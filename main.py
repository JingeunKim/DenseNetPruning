import shutil
import test
import utils
import dataloader
import torch
from GA import GA
import datetime
from utils import logger
import time
import train
import os
import torch.nn as nn
import torch.optim as optim

# start = time.time()
# utils.print_and_log(logger, "Data augmentation : {}".format(utils.augmentation))
#
# GA(nDenseBlock=100, Bottleneck=True).evolve()
# end = time.time()
# utils.print_and_log(logger, "GA TIME : {}".format(end-start))

# model = torch.load('./models/model{:%Y%m%d}_{}_{}.pt'.format(datetime.datetime.now(), utils.prob, str(utils.augmentation)))
model = torch.load('./models/model20231227_0.5_True.pt')

utils.print_and_log(logger, "Model : model{:%Y%m%d}_{}_{}.pt".format(datetime.datetime.now(), utils.prob, str(utils.augmentation)))
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
utils.print_and_log(logger, "train & test")
utils.print_and_log(logger, "Best model # params =  {}".format(num_params))
trainloader, testloader, classes = dataloader.dataloader()
model = train.train(model, trainloader, utils.epochs, utils.device)
acc = test.test(testloader, model, utils.device)

utils.print_and_log(logger, "Best model error rate =  {}%".format(acc))
utils.print_and_log(logger, "END")
