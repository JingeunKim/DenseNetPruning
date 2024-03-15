python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0
python main.py --dataset cifar-100 --nClasses 100 --augmentation True --dropout 0

#usage: main.py [-h] [--device DEVICE] [--dataset DATASET]
#               [--GA_epoch GA_EPOCH] [--crossover_rate CROSSOVER_RATE]
#               [--mutation_rate MUTATION_RATE] [--generation GENERATION]
#               [--number_population NUMBER_POPULATION] [--elitism ELITISM]
#               [--num_workers NUM_WORKERS] [--lr LR]
#               [--weight_decay WEIGHT_DECAY] [--momentum MOMENTUM]
#               [--epochs EPOCHS] [--growthRate GROWTHRATE] [--depth DEPTH]
#               [--reduction REDUCTION] [--bottleneck BOTTLENECK]
#               [--nClasses NCLASSES] [--augmentation AUGMENTATION]
#               [--dropout DROPOUT]
