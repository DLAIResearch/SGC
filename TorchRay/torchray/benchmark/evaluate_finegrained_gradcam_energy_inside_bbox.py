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
import resnet_multigpu_sgc as resnet
import cv2
import datasets as pointing_datasets
from collections import OrderedDict
""" 
    Here, we evaluate the content heatmap (Grad-CAM heatmap within object bounding box) on the fine-grained dataset.
"""

model_names = ['resnet18', 'resnet50']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-data', default='env/cub/',metavar='DIR', help='path to dataset')
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
parser.add_argument('--dataset', type=str, default='cub',
                            help='dataset to use: [imagenet2, cub, aircraft, flowers, cars]')


def main():
    global args
    args = parser.parse_args()

    if args.dataset == 'cub':
        num_classes = 200
    elif args.dataset == 'aircraft':
        num_classes = 90
    elif args.dataset == 'flowers':
        num_classes = 102
    elif args.dataset == 'cars':
        num_classes = 196

    print("=> creating model '{}' for '{}'".format(args.arch, args.dataset))
    if args.arch.startswith('resnet'):
        model = resnet.__dict__[args.arch](num_classes=num_classes)
    else:
        print('Other archs not supported')
        exit()
    model = torch.nn.DataParallel(model).cuda()
    # def copyStateDict(state_dict):
    #     if list(state_dict.keys())[0].startswith('module'):
    #         start_idx = 1
    #     else:
    #         start_idx = 0
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         name = '.'.join(k.split('.')[start_idx:])
    #         new_state_dict[name] = v
    #     return new_state_dict
    if args.resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        # if 'state_dict' in checkpoint:
        #     state_dict = checkpoint['state_dict']
            # new_dict = copyStateDict(state_dict)
            # keys = []
            # for k, v in new_dict.items():
            #     if k.startswith('conv_1.weight'):  # 将‘conv_cls’开头的key过滤掉，这里是要去除的层的key
            #         continue
            #     keys.append(k)
            #     # remove the module prefix if model was saved with DataParallel
            # new_dict = {k: new_dict[k] for k in keys}
            # new_dict = {k.replace('module.', ''): v for k, v in new_dict.items()}
            # # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # # state_dict = {k.startswith('conv_1.weight'): v for k, v in state_dict.items()}
            # # load params
            # model.load_state_dict(new_dict)


        # elif 'model' in checkpoint:
        #     model.load_state_dict(checkpoint['model'])
        # else:
        #     print('Checkpoint format not supported')
        #     exit()
    model = model.cuda()
    if (not args.resume) and (not args.pretrained):
        assert False, "Please specify either the pre-trained model or checkpoint for evaluation"

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    # In the version, we will not resize the images. We feed the full image and use AdaptivePooling before FC.
    # We will resize Gradcam heatmap to image size and compare the actual bbox co-ordinates
    val_dataset = pointing_datasets.ImageNetDetection(args.data,
                                                      transform=transforms.Compose([
                                                          transforms.Resize(args.input_resize),
                                                          transforms.ToTensor(),
                                                          normalize,
                                                        ]))

    # we set batch size=1 since we are loading full resolution images.
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    validate_multi(val_loader, val_dataset, model)


def validate_multi(val_loader, val_dataset, model):
    batch_time = AverageMeter()
    heatmap_inside_bbox = AverageMeter()

    # switch to evaluate mode


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
        model.eval()
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
