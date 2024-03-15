import shutil
import test
import utils
import dataloader
import torch
from GA import GA
import datetime
from utils import logger, arg
import time
import train
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
all_run = []
all_params = []
all_acc = []
for run in range(5):
    start = time.time()
    utils.print_and_log(logger, "Data augmentation : {}".format(arg.augmentation))

    GA(nDenseBlock=arg.depth, Bottleneck=arg.bottleneck).evolve()
    end = time.time()
    el_time = end-start
    utils.print_and_log(logger, "GA TIME : {}".format(el_time))
    all_run.append(el_time)
    model = torch.load('./models/model{:%Y%m%d}_{}.pt'.format(datetime.datetime.now(), str(arg.augmentation)))

    utils.print_and_log(logger, "Model : model{:%Y%m%d}_{}.pt".format(datetime.datetime.now(), str(arg.augmentation)))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params.append(num_params)
    utils.print_and_log(logger, "train & test")
    utils.print_and_log(logger, "Best model # params =  {}".format(num_params))
    trainloader, testloader, classes = dataloader.dataloader()
    model, best_error = train.train(model, trainloader, arg.epochs, arg.device, testloader)
    # acc = test.test(testloader, model, utils.device)
    utils.print_and_log(logger, "Best model error rate =  {}%".format(best_error))
    all_acc.append(best_error)
    utils.print_and_log(logger, "END")
    torch.save(model, './models/model_trained_{}_{}_{}.pt'.format(str(arg.dataset),
                                                                      str(arg.augmentation), str(run)))
    utils.print_and_log(logger, "Model saved")

utils.print_and_log(logger, "avg time =  {}, std = {}%".format(sum(all_run)/len(all_run), np.std(all_run)))
utils.print_and_log(logger, "avg error rate =  {}, std = {}%".format(sum(all_acc)/len(all_acc), np.std(all_acc)))
utils.print_and_log(logger, "avg # params =  {}, std = {}%".format(sum(all_params)/len(all_params), np.std(all_params)))