import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math
import random

"https://github.com/bamos/densenet.pytorch/blob/master/densenet.py"


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseBlock(nn.Module):
    def __init__(self, nChannels, growthRate, nDenseBlocks, matrix, idx):
        super(DenseBlock, self).__init__()
        self.nChannels = nChannels
        self.nChannels_copy = nChannels
        self.growthRate = growthRate
        self.layers = []
        self.matrix = matrix
        self.idx = idx
        self.nDenseBlocks = nDenseBlocks
        for i in range(1, nDenseBlocks):
            self.nChannels_copy = self.nChannels
            full_list = list(range(0, i))
            # 컬럼 돌며 0 숫자 카운트 (concat된 x 중 chunk하여 delete하기 위함)
            missing_numbers = [num for num in full_list if num not in self.idx[i]]
            if i > 1:
                for a in missing_numbers:
                    if a == 0:
                        self.nChannels_copy -= nChannels
                    else:
                        self.nChannels_copy -= 12
            interChannels = 4 * self.growthRate
            self.layers += [nn.BatchNorm2d(self.nChannels_copy),
                            nn.Conv2d(self.nChannels_copy, interChannels, kernel_size=1, bias=False),
                            nn.BatchNorm2d(interChannels),
                            nn.Conv2d(interChannels, self.growthRate, kernel_size=3, padding=1,
                                      bias=False)]
            self.nChannels += self.growthRate

        self.dense = nn.Sequential(*self.layers)

    def get_inputsize(self):
        return self.nChannels_copy + self.growthRate

    def forward(self, x):
        for i in range(1, len(self.dense) // 4 + 1):
            x_list = []
            if i == 1:  # 0번째 블락은 그냥 통과
                globals()["out{}".format(i - 1)] = self.dense[(i - 1) * 4:i * 4](x)
                globals()["x{}".format(i)] = torch.cat([x, globals()["out{}".format(i - 1)]], 1)
            else:
                for q in self.idx[i]:
                    if q == 0:
                        x_list.append(x)
                    else:
                        x_list.append(globals()["out{}".format(q - 1)])
                if x_list is None:
                    print(x_list)
                    print(self.idx[i])
                globals()["x{}".format(i - 1)] = torch.cat(x_list, 1)
                globals()["out{}".format(i - 1)] = self.dense[(i - 1) * 4:i * 4](globals()["x{}".format(i - 1)])
                globals()["x{}".format(i)] = torch.cat(
                    [globals()["x{}".format(i - 1)], globals()["out{}".format(i - 1)]], 1)

        return globals()["x{}".format(self.nDenseBlocks - 1)]



class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck, matrix, idx):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth - 4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        self.nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(3, self.nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = DenseBlock(self.nChannels, growthRate, nDenseBlocks, matrix[0], idx[0])
        self.nChannels = DenseBlock.get_inputsize(self.dense1)
        nOutChannels = int(math.floor(self.nChannels * reduction))
        self.trans1 = Transition(self.nChannels, nOutChannels)
        self.nChannels = int(math.floor(self.nChannels * reduction))

        self.dense2 = DenseBlock(self.nChannels, growthRate, nDenseBlocks, matrix[1], idx[1])
        self.nChannels = DenseBlock.get_inputsize(self.dense2)
        nOutChannels = int(math.floor(self.nChannels * reduction))

        self.trans2 = Transition(self.nChannels, nOutChannels)
        self.nChannels = int(math.floor(self.nChannels * reduction))

        self.dense3 = DenseBlock(self.nChannels, growthRate, nDenseBlocks, matrix[2], idx[2])
        self.nChannels = DenseBlock.get_inputsize(self.dense3)

        self.bn1 = nn.BatchNorm2d(self.nChannels)
        self.fc = nn.Linear(self.nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = out.view(-1, self.nChannels)
        return self.fc(out)
