#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).
from torchvision import transforms

MNIST_mean = (0.1307,)
MNIST_std = (0.3081,)
MNIST = [
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_mean, MNIST_std)
    ]),
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_mean, MNIST_std)
    ])
]

FashionMNIST_mean = (0.2860,)
FashionMNIST_std = (0.3205,)
FashionMNIST = [
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(FashionMNIST_mean, FashionMNIST_std)
    ]),
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(FashionMNIST_mean, FashionMNIST_std)
    ])
]

CIFAR10_mean = (0.49139968, 0.48215841, 0.44653091)
CIFAR10_std = (0.2023, 0.1994, 0.2010)
CIFAR10 = [
    transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_mean, CIFAR10_std)
    ]),
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_mean, CIFAR10_std)
    ])
]

CIFAR100_mean = (0.50707516, 0.48654887, 0.44091784)
CIFAR100_std = (0.26733429, 0.25643846, 0.27615047)
CIFAR100 = [
    transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_mean, CIFAR100_std),
    ]),
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_mean, CIFAR100_std),
    ])
]


def imagenet_like(resize_size=256, crop_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return [
        transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    ]


ImageNet_mean = (0.485, 0.456, 0.406)
ImageNet_std = (0.229, 0.224, 0.225)
ImageNet = imagenet_like()

T = {
    'mnist':         MNIST,
    'fashion-mnist': FashionMNIST,
    'cifar10':       CIFAR10,
    'cifar100':      CIFAR100,
    'imagenet':      ImageNet
}
