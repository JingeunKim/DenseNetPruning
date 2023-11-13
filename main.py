import matplotlib.pyplot as plt
import numpy as np
import torchvision
import utils
import dataloader

from model import DenseNet
from train import train
from test import test
import torch
from torchsummary import summary
from GA import GA
import datetime
from utils import logger
import time
start = time.time()
GA(nDenseBlock=100, Bottleneck=True).evolve()
end = time.time()
utils.print_and_log(logger, "GA TIME : {}".format(end-start))

model = torch.load('./models/model{:%Y%m%d}_{}.pt'.format(datetime.datetime.now(), utils.prob))
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
utils.print_and_log(logger, "train & test")
utils.print_and_log(logger, "Best model # params =  {}".format(num_params))
trainloader, testloader, classes = dataloader.dataloader()
model = train(model, trainloader, utils.epochs, utils.device)
acc = test(model, testloader, utils.device)
utils.print_and_log(logger, "Best model accuracy =  {}%".format(acc))
utils.print_and_log(logger, "END")

