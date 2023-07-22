# Enhancing Consistent Visual Explanations through Saliency-Guided Contrastive Learning
Deep neural networks have revolutionized computer vision tasks with their exceptional performance. However, their lack of interpretability has raised concerns and sparked the need for effective explanation techniques. While various methods have been proposed, recent investigations have revealed biases that result in explanations inconsistent with human knowledge. Our solution: a breakthrough approach that tackles this challenge head-on. Rather than proposing a new method of interpretation, this study instructs models on how to learn unbiased features that lead to explanations that are more consistent with the task and humans' priors, given existing methods of interpretation.

![Image text](https://github.com/DLAIResearch/SGC/misc/teaser_image.jpg)
<br/>
Code uploading in progress--
## Pre-requisites
- Pytorch 1.3 - Please install [PyTorch](https://pytorch.org/get-started/locally/) and CUDA if you don't have it installed.
- ## Datasets
 - [ImageNet - 100](https://www.image-net.org/download.php)
 - [CUB-200](https://vision.cornell.edu/se3/caltech-ucsd-birds-200/)
 - [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
 - [Stanford Cars-196](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
 - [VGG Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

## Training

#### Train and evaluate ResNet18 and ResNet50 model on the ImageNet dataset using our CGC loss
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_eval_SGC.py /datasets/imagenet -a resnet50 -p 100 -j 8 -b 192 --lr 0.01 --lambda 0.5 -t 0.5 --save_dir <SAVE_DIR> --log_dir <LOG_DIR>
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_eval_SGC.py /datasets/imagenet -a resnet50 -p 100 -j 8 -b 256 --lr 0.01 --lambda 0.5 -t 0.5 --save_dir <SAVE_DIR> --log_dir <LOG_DIR>

#### Train and evaluate a ResNet50 model on 1pc labeled subset of ImageNet dataset and the rest as unlabeled dataset. We initialize the model from SwAV
For the below command, <PATH_TO_SWAV_MODEL_PRETRAINED> can be downloaded from the github directory of SwAV - https://github.com/facebookresearch/swav
We use the model checkpoint listed on the first row (800 epochs, 75.3% ImageNet top-1 acc.) of the Model Zoo of the above repository.

```
CUDA_VISIBLE_DEVICES=0,1 python train_imagenet_1pc_swav_cgc_unlabeled.py <PATH_TO_1%_IMAGENET> -a resnet50 -b 128 -j 8 --lambda 0.25 -t 0.5 --epochs 50 --lr 0.02 --lr_last_layer 5 --resume <PATH_TO_SWAV_MODEL_PRETRAINED> --save_dir <SAVE_DIR> --log_dir <LOG_DIR> 2>&1 | tee <PATH_TO_CMD_LOG_FILE>
```
