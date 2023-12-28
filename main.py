import shutil

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
import os
import torch.nn as nn


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (datetime.datetime.now() + utils.dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (datetime.datetime.now() + utils.dataset) + 'model_best.pth.tar')

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
best_prec1=0
for epoch in range(utils.epochs):
    # adjust_learning_rate(optimizer, epoch)
    # train for one epoch
    model = train(model, trainloader, utils.epochs, utils.device)
    # evaluate on validation set
    acc = test(testloader, model, utils.device)

    # remember best prec@1 and save checkpoint
    is_best = acc > best_prec1
    best_prec1 = max(acc, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
    }, is_best)
print('Best accuracy: ', best_prec1)
# model = train(model, trainloader, utils.epochs, utils.device)
# acc = test(testloader, model, utils.device)

utils.print_and_log(logger, "Best model error rate =  {}%".format(best_prec1))
utils.print_and_log(logger, "END")
