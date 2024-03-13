import logging
import datetime

device = 'cuda'
dataset = 'cifar-10'

GA_epoch = 10
crossover_rate = 0.9
mutation_rate = 0.05
generation = 50
number_population = 40
prob = 0.6

crossover = 'row-col'
elitism = 0.2

num_workers = 4
lr = 1e-1
weight_decay = 1e-4
momentum = 0.9
epochs = 300

growthRate = 12
depth = 100
reduction = 0.5
bottleneck = True
if dataset == 'cifar-10':
    nClasses = 10
elif dataset == 'cifar-100':
    nClasses = 100

augmentation = True

if augmentation != False:
    pass
else:
    dropout = 0.2


def setup_logger():
    logger = logging.getLogger()
    log_path = './logs/{:%Y%m%d}_{}.log'.format(datetime.datetime.now(), dataset)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def bench_setup_logger():
    logger = logging.getLogger()
    log_path = './bench_logs/{:%Y%m%d}_{}.log'.format(datetime.datetime.now(), dataset)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def bench_print_and_log(logger, msg):
    # global logger
    print(msg)
    logger.info(msg)

def print_and_log(logger, msg):
    # global logger
    print(msg)
    logger.info(msg)

logger = setup_logger()
# bench_logger = bench_setup_logger()