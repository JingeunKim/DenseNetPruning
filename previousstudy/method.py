import random
from model import DenseNet
from train import GAtrain
from test import test
import utils
import dataloader
import numpy as np
import torch
import os
import datetime
from utils import bench_logger
import multiprocessing as mp
import joblib
import pandas as pd
# from catboost import CatBoostRegressr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import time
import matplotlib.pyplot as plt

class GA():
    def __init__(self, nDenseBlock, Bottleneck):
        self.pop_size = 30
        self.generations = 10
        self.nDenseBlock = nDenseBlock
        self.Bottleneck = Bottleneck
        self.prob = utils.prob
        self.number_blocks = 3

        utils.bench_print_and_log(bench_logger,
                            "GA start generation : " + str(self.generations) + " BC = " + str(utils.bottleneck) + " population : " + str(self.pop_size))

    def create_init_pop(self, nb_layers):
        population = []
        i = 0
        for p in range(self.pop_size):
            for q in range(self.number_blocks):
                population.append(self.create_matrix(nb_layers, i))
                i += 1
        return population

    def chk_diagonal(self, nb_layers, matrix):
        for j in range(self.number_blocks):
            matrix_ = matrix[j]
            for i in range(nb_layers + 1):
                if matrix_[i][i + 1] == 0:
                    matrix_[i][i + 1] = 1
        return matrix

    def create_matrix(self, nb_layers, seed):
        # random seed
        random.seed(seed)

        nb_layers = nb_layers + 2
        matrix = [[0 for _ in range(nb_layers)] for _ in range(nb_layers)]

        for i in range(nb_layers - 1):
            matrix[i][i + 1] = 1

        one_prob = [0.9, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
        for i in range(nb_layers):
            for j in range(i + 2, nb_layers):
                prob = random.random()
                # print(prob)
                if prob <= self.prob:
                    matrix[i][j] = 1
                else:
                    matrix[i][j] = 0
        # for row in matrix:
        #     print(row)
        # print()
        return matrix

    def chk(self, nb_layers, matrix):
        idx = []
        # print("chk")
        for j in range(nb_layers + 2):
            z = [i[j] for i in matrix]
            # print("z")
            # print(z)
            find_one = []
            for a in range(len(z)):
                if z[a] == 1:
                    find_one.append(a)
            # print("find_one")
            # print(find_one)
            idx.append(find_one)
        return idx

    def fitness(self, acc, params):
        fitness = [x / y for x, y in zip(acc, params)]
        return fitness

    def selection(self, selected_number, seed):
        random.seed(seed)
        parents = random.sample(selected_number, 2)
        for rand in parents:
            selected_number.remove(rand)
        return parents, selected_number

    def crossover(self, p1, p2, new_population, nDenseBlocks):
            # one-point crossover row
        parent1 = np.array(new_population[p1 * self.number_blocks:(p1 + 1) * self.number_blocks])
        parent2 = np.array(new_population[p2 * self.number_blocks:(p2 + 1) * self.number_blocks])

        point = np.random.randint(1, nDenseBlocks)
        offspring1 = np.concatenate((parent1[:, :point], parent2[:, point:]),
                                    axis=1)
        offspring2 = np.concatenate((parent2[:, :point], parent1[:, point:]),
                                    axis=1)

        offspring1 = offspring1.tolist()
        offspring2 = offspring2.tolist()
        return offspring1, offspring2

    def mutation(self, offspring1, offspring2, nDenseBlocks):
            # random point bit-flip  mutation
        off1 = np.array(offspring1)
        off2 = np.array(offspring2)

        row_index = np.random.randint(0, nDenseBlocks)

        col_index = np.random.randint(row_index + 1, nDenseBlocks + 1)

        for i in range(self.number_blocks):
            if off1[i][row_index][col_index] == 0:
                off1[i][row_index][col_index] = 1
            else:
                off1[i][row_index][col_index] = 0
        for i in range(self.number_blocks):
            if off2[i][row_index][col_index] == 0:
                off2[i][row_index][col_index] = 1
            else:
                off2[i][row_index][col_index] = 0

        offspring1 = off1.tolist()
        offspring2 = off2.tolist()

        return offspring1, offspring2


    def evolve(self):
        # if not os.path.exists('./previousstudy/bench_models'): os.mkdir('./previousstudy/bench_models')
        # if not os.path.exists('./previousstudy/bench_logs'): os.mkdir('./previousstudy/bench_logs')

        nDenseBlocks = (self.nDenseBlock - 4) // 3
        if self.Bottleneck:
            nDenseBlocks //= 2

        population = self.create_init_pop(nDenseBlocks)

        idx = []
        for p in range(self.pop_size * 3):
            idx.append(self.chk(nDenseBlocks, population[p]))

        trainloader, testloader, classes = dataloader.GAdataloader()
        acc = []

        for i in range(1, self.pop_size + 1):
            net = DenseNet(growthRate=utils.growthRate, depth=utils.depth, reduction=utils.reduction,
                           bottleneck=utils.bottleneck, nClasses=utils.nClasses,
                           matrix=population[(i - 1) * self.number_blocks:i * self.number_blocks],
                           idx=idx[(i - 1) * self.number_blocks:i * self.number_blocks]).to(
                device=utils.device)
            net, loss = GAtrain(net, trainloader, utils.GA_epoch, utils.device)
            acc.append(loss)

        utils.bench_print_and_log(bench_logger, "loss = {}".format(acc))
        fitness = acc


        new_population = population
        for generation in range(self.generations):
            utils.bench_print_and_log(bench_logger, "-----" + str(generation + 1) + " generation------")
            selected_number = list(range(self.pop_size))
            for n in range(len(new_population)//3//2):
                offspring1 = []
                offspring2 = []
                seed = np.random.rand()
                crossover_rand = np.random.rand()
                print("cross rand = ", crossover_rand)
                if crossover_rand <= 0.5:
                    parents, selected_number = self.selection(selected_number, seed)
                    offspring1, offspring2 = self.crossover(parents[0], parents[1], new_population, nDenseBlocks)
                    mutate_rand = np.random.rand()
                    if mutate_rand <= 0.05:
                        offspring1, offspring2 = self.mutation(offspring1, offspring2, nDenseBlocks)
                    offspring1 = self.chk_diagonal(nDenseBlocks, offspring1)
                    offspring2 = self.chk_diagonal(nDenseBlocks, offspring2)
                    new_population.extend(offspring1)
                    new_population.extend(offspring2)
            else:
                pass
            for p in range(self.pop_size + 1, len(new_population)):
                idx.append(self.chk(nDenseBlocks, new_population[p]))

            for m in range(self.pop_size, len(new_population)//3):
                net = DenseNet(growthRate=utils.growthRate, depth=utils.depth, reduction=utils.reduction,
                               bottleneck=utils.bottleneck, nClasses=utils.nClasses,
                               matrix=new_population[(m - 1) * self.number_blocks:(m) * self.number_blocks],
                               idx=idx[((m) - 1) * self.number_blocks:(m) * self.number_blocks]).to(
                    device=utils.device)
                net, loss = GAtrain(net, trainloader, utils.GA_epoch, utils.device)
                acc.append(loss)

            utils.bench_print_and_log(bench_logger, "loss = {}".format(acc))
            fitness = acc

            parents_population = new_population[:self.pop_size * self.number_blocks]
            parents_fitness = fitness[:self.pop_size]
            idx_parents = idx[:self.pop_size * self.number_blocks]
            parents_acc = acc[:self.pop_size]

            offspring_population = new_population[self.pop_size * self.number_blocks:]
            offspring_fitness = fitness[self.pop_size:]
            idx_offspring = idx[self.pop_size * self.number_blocks:]
            offspring_acc = acc[self.pop_size:]

            parent_rank = np.argsort(parents_fitness)
            parents_population_rank = []
            idx_parents_rank = []
            for i in parent_rank + 1:
                parents_population_rank.extend(parents_population[(i - 1) * self.number_blocks:i * self.number_blocks])
                idx_parents_rank.extend(idx_parents[(i - 1) * self.number_blocks:i * self.number_blocks])
            parents_fitness = [parents_fitness[i] for i in parent_rank]
            # parents_second_fitness = [parents_second_fitness[i] for i in parent_rank]
            parents_acc = [parents_acc[i] for i in parent_rank]
            # parents_params = [parents_params[i] for i in parent_rank]

            offspring_rank = np.argsort(offspring_fitness)
            offspring_population_rank = []
            idx_offspring_rank = []
            for i in offspring_rank + 1:
                offspring_population_rank.extend(
                    offspring_population[(i - 1) * self.number_blocks:i * self.number_blocks])
                idx_offspring_rank.extend(idx_offspring[(i - 1) * self.number_blocks:i * self.number_blocks])
            offspring_fitness = [offspring_fitness[i] for i in offspring_rank]
            # offspring_second_fitness = [offspring_second_fitness[i] for i in offspring_rank]
            offspring_acc = [offspring_acc[i] for i in offspring_rank]
            # offspring_params = [offspring_params[i] for i in offspring_rank]

            elite_rate = 4
            parents_population = parents_population_rank[:int(self.pop_size * elite_rate) * self.number_blocks]
            parents_fitness = parents_fitness[:int(self.pop_size * elite_rate)]
            # parents_second_fitness = parents_second_fitness[:int(self.pop_size * elite_rate)]
            idx_parents = idx_parents_rank[:int(self.pop_size * elite_rate) * self.number_blocks]
            parents_acc = parents_acc[:int(self.pop_size * elite_rate)]
            # parents_params = parents_params[:int(self.pop_size*elite_rate)]
            off_elite_rate=0.9
            offspring_population = offspring_population_rank[:int(self.pop_size * (1 - off_elite_rate)) * self.number_blocks]
            offspring_fitness = offspring_fitness[:int(self.pop_size * (1 - off_elite_rate))]
            # offspring_second_fitness = offspring_second_fitness[:int(self.pop_size * (1 - elite_rate))]
            idx_offspring = idx_offspring_rank[:int(self.pop_size * (1 - off_elite_rate)) * self.number_blocks]
            offspring_acc = offspring_acc[:int(self.pop_size * (1 - off_elite_rate))]
            # offspring_params = offspring_params[:int(self.pop_size*elite_rate)]
            new_population = parents_population + offspring_population  # np.concatenate((parents_population, offspring_population), axis=0)
            fitness = parents_fitness + offspring_fitness
            # second_fitness = parents_second_fitness + offspring_second_fitness
            idx = idx_parents + idx_offspring
            acc = parents_acc + offspring_acc
            # params = parents_params + offspring_params

            rank = np.argsort(fitness)
            utils.bench_print_and_log(bench_logger, "Fitness values = {}".format(fitness))
            best_fitness = [fitness[i] for i in rank]
            utils.bench_print_and_log(bench_logger, "Best Fitness value = {}".format(best_fitness[0]))

        rank = np.argsort(fitness)
        population_rank = []
        idx_rank = []
        for i in rank + 1:
            population_rank.extend(new_population[(i - 1) * self.number_blocks:i * self.number_blocks])
            idx_rank.extend(idx[(i - 1) * self.number_blocks:i * self.number_blocks])

        idx = idx[:self.number_blocks]
        best_model = new_population[:self.number_blocks]

        net = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=utils.nClasses,
                       matrix=best_model[0:self.number_blocks],
                       idx=idx[0:self.number_blocks]).to(
            device=utils.device)
        print(net)
        torch.save(net, './bench_models/model{:%Y%m%d}_{}.pt'.format(datetime.datetime.now(),
                                                                  str(utils.augmentation)))
        utils.bench_print_and_log(bench_logger, "# Params = {}".format(sum(p.numel() for p in net.parameters() if p.requires_grad)))
        utils.bench_print_and_log(bench_logger, "Finish")
