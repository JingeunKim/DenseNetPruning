import torch
import torchvision
import torchvision.transforms as transforms
import utils
import numpy as np
from utils import arg
from torch.utils.data import SubsetRandomSampler
def dataloader():
    if arg.dataset =='cifar-10':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    elif arg.dataset =='cifar-100':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [129.3, 124.1, 112.4]],
                                         std=[x / 255.0 for x in [68.2, 65.4, 70.4]])
    elif arg.dataset == 'ImageNet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    if arg.dataset != 'ImageNet':
        if arg.augmentation == 'True':
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize])
    else:
        transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             normalize,
             ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    batch_size = 64
    num_workers = arg.num_workers

    if arg.dataset == 'cifar-10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_test)
        classes = 10

    elif arg.dataset == 'cifar-100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=transform)

        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                               download=True, transform=transform_test)
        classes=100

    elif arg.dataset == 'ImageNet':
        trainset = torchvision.datasets.ImageNet(root='./data', split='train',
                                                download=True, transform=transform)

        testset = torchvision.datasets.ImageNet(root='./data', split='val',
                                               download=True, transform=transform_test)
        classes = 1000

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    return trainloader, testloader, classes


def GAdataloader():
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    if arg.dataset =='cifar-10':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    elif arg.dataset =='cifar-100':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [129.3, 124.1, 112.4]],
                                         std=[x / 255.0 for x in [68.2, 65.4, 70.4]])
    elif arg.dataset == 'ImageNet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    if arg.dataset != 'ImageNet':
        if arg.augmentation == 'True':
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize])
    else:
        transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             normalize,
             ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    val_rate = arg.subset
    batch_size = 64
    num_workers = arg.num_workers

    if arg.dataset == 'cifar-10':
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform)

        len_trainset = len(train_set)
        indices = list(range(len_trainset))
        split_point = int(np.floor(len_trainset * val_rate))
        train_idx, valid_idx = indices[split_point:], indices[:split_point]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        classes = 10
    elif arg.dataset == 'cifar-100':
        train_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                 download=True, transform=transform)
        len_trainset = len(train_set)
        indices = list(range(len_trainset))
        split_point = int(np.floor(len_trainset * val_rate))
        train_idx, valid_idx = indices[split_point:], indices[:split_point]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        classes = 100
    elif arg.dataset == 'ImageNet':
        train_set = torchvision.datasets.ImageNet(root='./data', split='train', transform=transform)
        len_trainset = len(train_set)
        indices = list(range(len_trainset))
        split_point = int(np.floor(len_trainset * val_rate))
        train_idx, valid_idx = indices[split_point:], indices[:split_point]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        classes = 1000

    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers
    )
    testloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers
    )
    return trainloader, testloader, classes