import shutil
import test
import utils
import dataloader
import torch
import datetime
from utils import bench_logger
import time
import train
import os
import torch.nn as nn
import torch.optim as optim
from method import GA
start = time.time()
utils.print_and_log(bench_logger, "Data augmentation : {}".format(utils.augmentation))

GA(nDenseBlock=utils.depth, Bottleneck=utils.bottleneck).evolve()
end = time.time()
utils.print_and_log(bench_logger, "GA TIME : {}".format(end-start))

model = torch.load('./bench_models/model{:%Y%m%d}_{}.pt'.format(datetime.datetime.now(), str(utils.augmentation)))

utils.print_and_log(bench_logger, "Model : model{:%Y%m%d}_{}.pt".format(datetime.datetime.now(), str(utils.augmentation)))
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
utils.print_and_log(bench_logger, "train & test")
utils.print_and_log(bench_logger, "Best model # params =  {}".format(num_params))
trainloader, testloader, classes = dataloader.dataloader()
model, best_error = train.train(model, trainloader, utils.epochs, utils.device, testloader)
# acc = test.test(testloader, model, utils.device)

utils.print_and_log(bench_logger, "Best model error rate =  {}%".format(best_error))
utils.print_and_log(bench_logger, "END")
