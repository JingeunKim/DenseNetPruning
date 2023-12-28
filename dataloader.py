import torch
import torchvision
import torchvision.transforms as transforms
import utils
import numpy as np

from torch.utils.data import SubsetRandomSampler
def dataloader():
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if utils.augmentation:
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
    num_workers = utils.num_workers

    if utils.dataset == 'cifar-10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_test)
        classes = 10

    elif utils.dataset == 'cifar-100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=transform)

        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                               download=True, transform=transform_test)
        classes=100

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)
    # 클래스들
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes


def GAdataloader():
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    if utils.augmentation:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    val_rate = 0.8
    batch_size = 64
    num_workers = utils.num_workers
    if utils.dataset == 'cifar-10':
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
    else:
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

    return trainloader, testloader, classes