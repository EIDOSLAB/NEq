[![paper](https://img.shields.io/badge/BMVC-paper-blue)](https://openreview.net/pdf?id=LGDfv0U7MJR)

# To update or not to update? Neurons at equilibrium in deep models - NeurIPS 2022

Official repository for NEq as presented at the NeurIPS 2022 conference.

## Requirements

Python==3.9 \
tqdm==4.62.3 \
numpy==1.19.5 \
filelock==3.2.0 \
torch==1.10.1 \
torchvision==0.11.2

## Reproduce our results

### CIFAR-10 - ResNet-32

`python3 train_classification.py --amp=1 --arch=resnet32-cifar --batch-size=100 --dataset=cifar10 --device=cuda --epochs=250 --eps=0.001 --lr=0.1 --momentum=0.9 --optim=sgd --val-size=0.01 --velocity-mu=0.5 --weight-decay=0.0005`

### ImageNet - ResNet-18

`python3 train_classification.py --amp=1 --arch=resnet18-imagenet --batch-size=128 --dataset=imagenet --device=cuda --epochs=90 --eps=0.001 --lr=0.1 --momentum=0.9 --optim=sgd --val-size=0.001 --velocity-mu=0.5 --weight-decay=0.0005`

### ImageNet - SwinB

From inside the SwinTransformer sub-dir

`python3 main.py --local_rank=0 --cfg configs/swin_base_patch4_window7_224_22kto1k_finetune.yaml --batch-size 64 --accumulation-steps 2 --val-size 0.001 --eps 1e-3 --velocity-mu 0.5`

### COCO - Deeplabv3

`python3 train_segmentation.py --lr 0.02 --dataset coco --arch deeplabv3_resnet50 --aux-loss --weights-backbone ResNet50_Weights.IMAGENET1K_V1 --val-size 0.001 --eps 2e-2 --batch-size 32 --amp --velocity-mu 0.5`

# Citation
Please cite this work as
```
@inproceedings{
bragagnolo2022to,
title={To update or not to update? Neurons at equilibrium in deep models},
author={Andrea Bragagnolo and Enzo Tartaglione and Marco Grangetto},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=LGDfv0U7MJR}
}
```
