import argparse
import logging
import datetime

parser = argparse.ArgumentParser(description="parameter of SMA-GA")
parser.add_argument("--device", type=str, default="cuda", help="type of gpu")
parser.add_argument("--dataset", type=str, default="cifar-10", help="cifar-10 or cifar-100")
parser.add_argument("--GA_epoch", type=int, default=10, help="number of epochs in GA")
parser.add_argument("--crossover_rate", type=float, default=1, help="probability of crossover")
parser.add_argument("--mutation_rate", type=float, default=0.05, help="probability of mutation")
parser.add_argument("--generation", type=int, default=60, help="number of generation")
parser.add_argument("--number_population", type=int, default=40, help="number_population")
parser.add_argument("--elitism", type=float, default=0.2, help="elitism rate")
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
parser.add_argument("--lr", type=float, default=1e-1, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight_decay")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--epochs", type=int, default=300, help="epochs")
parser.add_argument("--growthRate", type=int, default=12, help="growthRate")
parser.add_argument("--depth", type=int, default=100, help="depth")
parser.add_argument("--reduction", type=float, default=0.5, help="reduction")
parser.add_argument("--bottleneck", type=bool, default=True, help="bottleneck")
parser.add_argument("--nClasses", type=int, default=10, help="number of classes")
parser.add_argument("--augmentation", type=str, default="True", help="augmentation")
parser.add_argument("--surrogate", type=str, default="True", help="surrogate")
parser.add_argument("--subset", type=float, default=0.9, help="ratio subset of training data")
parser.add_argument("--dropout", type=float, default=0., help="set 0.2 if you do not data aumentation")
parser.add_argument("--run", type=int, default=5, help="run")
arg = parser.parse_args()
print(arg)
def setup_logger():
    logger = logging.getLogger()
    log_path = './logs/{:%Y%m%d}_{}.log'.format(datetime.datetime.now(), arg.dataset)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def bench_setup_logger():
    logger = logging.getLogger()
    log_path = './bench_logs/{:%Y%m%d}_{}.log'.format(datetime.datetime.now(), arg.dataset)
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