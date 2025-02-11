#python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0 --number_mutation 1 --number_crossover 1 --run 4
#python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0 --number_mutation 2 --number_crossover 2 --run 4
##
##
#python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0 --number_population 20 --generation 60 --run 4
#python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0 --number_population 20 --generation 40 --run 4
#python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0 --number_population 40 --generation 40 --run 4
#
#
#python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0 --number_population 40 --generation 60 --run 2
#python main.py --dataset cifar-100 --nClasses 100 --augmentation True --dropout 0 --number_population 40 --generation 60 --run 2
#python main.py --dataset cifar-100 --nClasses 100 --augmentation False --dropout 0 --number_population 40 --generation 60 --run 2
#
#python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0 --number_population 40 --generation 60 --run 5
#python main.py --dataset cifar-100 --nClasses 100 --augmentation True --dropout 0 --number_population 40 --generation 60 --run 5
#python main.py --dataset cifar-100 --nClasses 100 --augmentation False --dropout 0 --number_population 40 --generation 60 --run 5
#python main.py --dataset cifar-100 --nClasses 100 --augmentation True --dropout 0 --number_population 40 --generation 60 --run 5

#python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0 --number_population 40 --generation 60 --crossover_rate 0.9 --mutation_rate 0.05 --run 1
#python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0 --number_population 40 --generation 60 --crossover_rate 0.9 --mutation_rate 0.1 --run 1
python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0 --number_population 40 --generation 60 --crossover_rate 0.9 --mutation_rate 0.2 --run 1


python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0 --number_population 40 --generation 60 --crossover_rate 0.8 --mutation_rate 0.05 --run 1
python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0 --number_population 40 --generation 60 --crossover_rate 0.8 --mutation_rate 0.1 --run 1
python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0 --number_population 40 --generation 60 --crossover_rate 0.8 --mutation_rate 0.2 --run 1


python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0 --number_population 40 --generation 60 --crossover_rate 1 --mutation_rate 0.1 --run 1
python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0 --number_population 40 --generation 60 --crossover_rate 1 --mutation_rate 0.2 --run 1

#python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0 --mode rs
#python main.py --dataset cifar-10 --nClasses 10 --augmentation False --dropout 0 --mode rs
#python main.py --dataset cifar-100 --nClasses 100 --augmentation True --dropout 0 --mode rs
#python main.py --dataset cifar-100 --nClasses 100 --augmentation False --dropout 0 --mode rs


#python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0.2
#python main.py --dataset cifar-100 --nClasses 100 --augmentation False --dropout 0.2

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

#python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0 --surrogate True --dataset 0.9 --GA_epoch 10 --run 1
#python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0 --surrogate True --subset 0.8 --GA_epoch 10 --run 4
#python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0 --surrogate True --subset 0.7 --GA_epoch 10 --run 4
#python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0 --surrogate True --subset 0.9 --GA_epoch 30 --run 4
#python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0 --surrogate True --subset 0.8 --GA_epoch 30 --run 4
#python main.py --dataset cifar-10 --nClasses 10 --augmentation True --dropout 0 --surrogate True --subset 0.7 --GA_epoch 30 --run 4