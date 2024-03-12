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

start = time.time()
utils.print_and_log(logger, "Data augmentation : {}".format(utils.augmentation))

GA(nDenseBlock=utils.depth, Bottleneck=utils.bottleneck).evolve()
end = time.time()
utils.print_and_log(logger, "GA TIME : {}".format(end-start))

model = torch.load('./models/model{:%Y%m%d}_{}.pt'.format(datetime.datetime.now(), str(utils.augmentation)))

utils.print_and_log(logger, "Model : model{:%Y%m%d}_{}.pt".format(datetime.datetime.now(), str(utils.augmentation)))
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
utils.print_and_log(logger, "train & test")
utils.print_and_log(logger, "Best model # params =  {}".format(num_params))
trainloader, testloader, classes = dataloader.dataloader()
model, best_error = train.train(model, trainloader, utils.epochs, utils.device, testloader)
# acc = test.test(testloader, model, utils.device)
torch.save(model, './models/model_trained_{:%Y%m%d}_{}.pt'.format(datetime.datetime.now(),
                                                                  str(utils.augmentation)))
utils.print_and_log(logger, "Best model error rate =  {}%".format(best_error))
utils.print_and_log(logger, "END")
