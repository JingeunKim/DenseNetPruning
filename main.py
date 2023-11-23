import utils
import dataloader
from train import train
from test import test
import torch
from ptflops import get_model_complexity_info
from GA import GA
import datetime
from utils import logger
import time
import re

start = time.time()
utils.print_and_log(logger, "Data augmentation : {}".format(utils.augmentation))

GA(nDenseBlock=100, Bottleneck=True).evolve()
end = time.time()
utils.print_and_log(logger, "GA TIME : {}".format(end-start))

model = torch.load('./models/model{:%Y%m%d}_{}_{}.pt'.format(datetime.datetime.now(), utils.prob, str(utils.augmentation)))

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
utils.print_and_log(logger, "train & test")
utils.print_and_log(logger, "Best model # params =  {}".format(num_params))
trainloader, testloader, classes = dataloader.dataloader()
model = train(model, trainloader, utils.epochs, utils.device)
acc = test(model, testloader, utils.device)
utils.print_and_log(logger, "Best model accuracy =  {}%".format(acc))
utils.print_and_log(logger, "END")

