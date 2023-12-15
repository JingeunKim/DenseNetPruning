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
from multiprocessing import Pool
import os


# def run_ga(prob):
#     ga = GA(100, True, prob)
#     start_time = time.time()
#     net = ga.evolve()
#     end_time = time.time()
#     return prob, net, end_time - start_time
#
# if __name__ == '__main__':
#     pool = Pool(os.cpu_count())
#
#     prob_values = [utils.prob, utils.prob + 0.1, utils.prob + 0.2, utils.prob + 0.3, utils.prob + 0.4]
#
#     results = pool.map_async(run_ga, prob_values)
#     result_get = results.get()
#
#     for prob, result, time_taken in result_get:
#         utils.print_and_log(logger, f"prob = {prob} acc = {result} GA TIME : {time_taken}")
#
#     for p in results:
#         pool.close()
#
#     pool.close()
#     pool.join()


start = time.time()
utils.print_and_log(logger, "Data augmentation : {}".format(utils.augmentation))

GA(nDenseBlock=100, Bottleneck=True).evolve()
end = time.time()
utils.print_and_log(logger, "GA TIME : {}".format(end-start))

model = torch.load('./models/model{:%Y%m%d}_{}_{}.pt'.format(datetime.datetime.now(), utils.prob, str(utils.augmentation)))

utils.print_and_log(logger, "Model : model{:%Y%m%d}_{}_{}.pt".format(datetime.datetime.now(), utils.prob, str(utils.augmentation)))
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
utils.print_and_log(logger, "train & test")
utils.print_and_log(logger, "Best model # params =  {}".format(num_params))
trainloader, testloader, classes = dataloader.dataloader()
model = train(model, trainloader, utils.epochs, utils.device)
acc = test(model, testloader, utils.device)
utils.print_and_log(logger, "Best model accuracy =  {}%".format(acc))
utils.print_and_log(logger, "END")

