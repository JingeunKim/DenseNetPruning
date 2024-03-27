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
    elif arg.dataset =='shvn':
        normalize = transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                         std=[0.24703223, 0.24348513, 0.26158784])

    if arg.augmentation:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

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

    elif arg.dataset == 'shvn':
        trainset = torchvision.datasets.SVHN(root='./data', split='train',
                                                download=True, transform=transform)

        testset = torchvision.datasets.SVHN(root='./data', split='test',
                                               download=True, transform=transform_test)
        classes=10

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    return trainloader, testloader, classes


def GAdataloader():
    if arg.dataset =='cifar-10':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    elif arg.dataset =='cifar-100':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [129.3, 124.1, 112.4]],
                                         std=[x / 255.0 for x in [68.2, 65.4, 70.4]])
    elif arg.dataset =='shvn':
        normalize = transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                         std=[0.24703223, 0.24348513, 0.26158784])
    if arg.augmentation:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    val_rate = 0.9
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
        trainloader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers
        )
        testloader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers
        )
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
        trainloader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers
        )
        testloader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers
        )
        classes = 100
    elif arg.dataset == 'shvn':
        train_set = torchvision.datasets.SVHN(root='./data', split='train',
                                                 download=True, transform=transform)
        len_trainset = len(train_set)
        indices = list(range(len_trainset))
        split_point = int(np.floor(len_trainset * val_rate))
        train_idx, valid_idx = indices[split_point:], indices[:split_point]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        trainloader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers
        )
        testloader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers
        )
        classes = 10
    return trainloader, testloader, classes