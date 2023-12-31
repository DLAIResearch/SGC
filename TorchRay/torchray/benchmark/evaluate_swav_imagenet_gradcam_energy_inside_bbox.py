import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
import swav_resnet_sgc as resnet
import os
import cv2
import datasets as pointing_datasets


""" 
    Here, we evaluate the content heatmap (Grad-CAM heatmap within object bounding box) on the imagenet2 dataset.
"""

model_names = ['resnet18', 'resnet50']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-data', metavar='DIR', default='',help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 96)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-g', '--num-gpus', default=1, type=int,
                    metavar='N', help='number of GPUs to match (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--input_resize', default=224, type=int,
                    metavar='N', help='Resize for smallest side of input (default: 224)')


def main():
    global args
    args = parser.parse_args()

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        if args.arch.startswith('resnet'):
            model = resnet.__dict__[args.arch](pretrained=True)
        else:
            assert False, 'Unsupported architecture: {}'.format(args.arch)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch.startswith('resnet'):
            model = resnet.__dict__[args.arch]()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            state_dict = torch.load(args.resume)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # pdb.set_trace()
            for k, v in model.state_dict().items():
                if k not in list(state_dict):
                    print('key "{}" could not be found in provided state dict'.format(k))
                elif state_dict[k].shape != v.shape:
                    print('key "{}" is of different shape in model and provided state dict'.format(k))
                    state_dict[k] = v
            model.load_state_dict(state_dict, strict=False)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    model = torch.nn.DataParallel(model).cuda()

    if (not args.resume) and (not args.pretrained):
        assert False, "Please specify either the pre-trained model or checkpoint for evaluation"

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    val_dataset = pointing_datasets.ImageNetDetection(args.data,
                                                      transform=transforms.Compose([
                                                          transforms.Resize(args.input_resize),
                                                          transforms.ToTensor(),
                                                          normalize,
                                                        ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    validate_multi(val_loader, val_dataset, model)


def validate_multi(val_loader, val_dataset, model):
    batch_time = AverageMeter()
    heatmap_inside_bbox = AverageMeter()

    # switch to evaluate mode
    model.eval()

    zero_count = 0
    total_count = 0
    end = time.time()
    for i, (images, annotation, targets) in enumerate(val_loader):
        total_count += 1
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # we assume batch size == 1 and unwrap the first elem of every list in annotation object
        annotation = unwrap_dict(annotation)
        image_size = val_dataset.as_image_size(annotation)

        output, feats = model(images, vanilla_with_feats=True)
        output_gradcam = compute_gradcam(output, feats, targets)
        output_gradcam_np = output_gradcam.data.cpu().numpy()[0]    # since we have batch size==1
        resized_output_gradcam = cv2.resize(output_gradcam_np, image_size)
        spatial_sum = resized_output_gradcam.sum()
        if spatial_sum <= 0:
            zero_count += 1
            continue

        # resized_output_gradcam is now normalized and can be considered as probabilities
        resized_output_gradcam = resized_output_gradcam / spatial_sum

        mask = pointing_datasets.imagenet_as_mask(annotation, targets[0].item())

        mask = mask.type(torch.ByteTensor)
        mask = mask.cpu().data.numpy()

        gcam_inside_gt_mask = mask * resized_output_gradcam
        # Now we sum the heatmap inside the object bounding box
        total_gcam_inside_gt_mask = gcam_inside_gt_mask.sum()
        heatmap_inside_bbox.update(total_gcam_inside_gt_mask)

        if i % 1000 == 0:
            print('\nResults after {} examples: '.format(i+1))
            print('Curr % of heatmap inside bbox: {:.4f} ({:.4f})'.format(heatmap_inside_bbox.val * 100,
                                                                             heatmap_inside_bbox.avg * 100))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('\nFinal Results - ')
    print('\n\n% of heatmap inside bbox: {:.4f}'.format(heatmap_inside_bbox.avg * 100))
    print('Zero GC found for {}/{} samples'.format(zero_count, total_count))

    return


def compute_gradcam(output, feats, target):
    """
    Compute the gradcam for the top predicted category
    :param output:
    :param feats:
    :return:
    """
    eps = 1e-8
    relu = nn.ReLU(inplace=True)

    target = target.cpu().numpy()
    # target = np.argmax(output.cpu().data.numpy(), axis=-1)
    one_hot = np.zeros((output.shape[0], output.shape[-1]), dtype=np.float32)
    indices_range = np.arange(output.shape[0])
    one_hot[indices_range, target[indices_range]] = 1
    one_hot = torch.from_numpy(one_hot)
    one_hot.requires_grad = True

    # Compute the Grad-CAM for the original image
    one_hot_cuda = torch.sum(one_hot.cuda() * output)
    dy_dz1, = torch.autograd.grad(one_hot_cuda, feats, grad_outputs=torch.ones(one_hot_cuda.size()).cuda(),
                                  retain_graph=True, create_graph=True)
    dy_dz_sum1 = dy_dz1.sum(dim=2).sum(dim=2)
    gcam512_1 = dy_dz_sum1.unsqueeze(-1).unsqueeze(-1) * feats
    # Comment the above 2 lines and uncomment the below one to change to dot product of grad and features to preserve grad spatial locations
    # gcam512_1 = dy_dz1 * feats
    gradcam = gcam512_1.sum(dim=1)
    gradcam = relu(gradcam)
    spatial_sum1 = gradcam.sum(dim=[1, 2]).unsqueeze(-1).unsqueeze(-1)
    gradcam = (gradcam / (spatial_sum1 + eps)) + eps

    return gradcam


def unwrap_dict(dict_object):
    new_dict = {}
    for k, v in dict_object.items():
        if k == 'object':
            new_v_list = []
            for elem in v:
                new_v_list.append(unwrap_dict(elem))
            new_dict[k] = new_v_list
            continue
        if isinstance(v, dict):
            new_v = unwrap_dict(v)
        elif isinstance(v, list) and len(v) == 1:
            new_v = v[0]
            # if isinstance(new_v, dict):
            #     new_v = unwrap_dict(new_v)
        else:
            new_v = v
        new_dict[k] = new_v
    return new_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
