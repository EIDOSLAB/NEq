# NEq source code

Implementation of NEq for CIFAR-10 and ImageNet classification tasks.

Here we provide instructions on how to replicate the reported experiments for ResNet-32 trained on CIFAR-10 and
ResNet-18 trained on ImageNet.

## Requirements

Python==3.9 \
tqdm==4.62.3 \
numpy==1.19.5 \
filelock==3.2.0 \
torch==1.10.1 \
torchvision==0.11.2

## CIFAR-10

`python3 train_classification.py --amp=1 --arch=resnet32-cifar --batch-size=100 --dataset=cifar10 --device=cuda --epochs=250 --eps=0.001 --lr=0.1 --momentum=0.9 --optim=sgd --val-size=0.01 --velocity-mu=0.5 --weight-decay=0.0005`

## ImageNet

`python3 train_classification.py --amp=1 --arch=resnet18-imagenet --batch-size=128 --dataset=imagenet --device=cuda --epochs=90 --eps=0.001 --lr=0.1 --momentum=0.9 --optim=sgd --val-size=0.001 --velocity-mu=0.5 --weight-decay=0.0005`