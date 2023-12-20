import random
from model import DenseNet
from train import train
from test import test
import utils
import dataloader
import numpy as np
import torch
import os
import datetime
from utils import logger
import multiprocessing as mp
import joblib
import pandas as pd
# from catboost import CatBoostRegressr
from sklearn.ensemble import GradientBoostingRegressor
import time

class GA():
    def __init__(self, nDenseBlock, Bottleneck):
        self.pop_size = utils.number_population
        self.generations = utils.generation
        self.nDenseBlock = nDenseBlock
        self.Bottleneck = Bottleneck
        self.prob = utils.prob
        self.number_blocks = 3

        utils.print_and_log(logger,
                            "GA start generation : " + str(self.generations) + " population : " + str(
                                self.pop_size) + " prob : " + str(self.prob) + " crossover : " + str(utils.crossover))

    def create_init_pop(self, nb_layers):
        population = []
        for p in range(self.pop_size):
            for q in range(self.number_blocks):
                population.append(self.create_matrix(nb_layers, p))
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
        # 2 차원 배열 초기화
        matrix = [[0 for _ in range(nb_layers)] for _ in range(nb_layers)]

        # 대각선 위 요소를 1로 채우기
        for i in range(nb_layers - 1):
            matrix[i][i + 1] = 1

        # 대각선 위 요소 이외의 요소를 랜덤하게 0 또는 1로 채우기
        for i in range(nb_layers):
            for j in range(i + 2, nb_layers):
                prob = random.random()
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

    def selection(self, selected_number):
        parents = random.sample(selected_number, 2)
        for rand in parents:
            selected_number.remove(rand)

        return parents, selected_number

    def crossover(self, p1, p2, new_population, nDenseBlocks):
        if utils.crossover == 'one-point':
            # one-point crossover
            parent1 = np.array(new_population[p1 * self.number_blocks:(p1 + 1) * self.number_blocks])
            parent2 = np.array(new_population[p2 * self.number_blocks:(p2 + 1) * self.number_blocks])

            point = random.randint(1, nDenseBlocks + 2)

            offspring1 = np.concatenate((parent1[:, :point], parent2[:, point:]),
                                        axis=1)
            offspring2 = np.concatenate((parent2[:, :point], parent1[:, point:]),
                                        axis=1)
            offspring1 = offspring1.tolist()
            offspring2 = offspring2.tolist()
        elif utils.crossover == 'row-col':
            parent1 = np.array(new_population[p1 * self.number_blocks:(p1 + 1) * self.number_blocks])
            parent2 = np.array(new_population[p2 * self.number_blocks:(p2 + 1) * self.number_blocks])

            row_point = random.randint(1, nDenseBlocks + 1)
            col_point = random.randint(1, nDenseBlocks + 1)

            offspring1_part = np.concatenate((parent1[:, :row_point, :col_point], parent2[:, :row_point, col_point:]),
                                             axis=2)
            offpsring1_part2 = np.concatenate((parent2[:, row_point:, :col_point], parent1[:, row_point:, col_point:]),
                                              axis=2)

            offspring1 = np.concatenate((offspring1_part, offpsring1_part2), axis=1)

            offspring2_part = np.concatenate((parent2[:, :row_point, :col_point], parent1[:, :row_point, col_point:]),
                                             axis=2)
            offpsring2_part2 = np.concatenate((parent1[:, row_point:, :col_point], parent2[:, row_point:, col_point:]),
                                              axis=2)

            offspring2 = np.concatenate((offspring2_part, offpsring2_part2), axis=1)

        return offspring1, offspring2

    def mutation(self, offspring1, offspring2, nDenseBlocks):
        # random point bit-flip mutation
        off1 = np.array(offspring1)
        off2 = np.array(offspring2)

        row_index = random.randint(0, nDenseBlocks)

        col_index = random.randint(row_index + 1, nDenseBlocks + 1)

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
        c_proc = mp.current_process()

        print("Running on Process", c_proc.name, "PID", c_proc.pid)
        if not os.path.exists('./models'): os.mkdir('./models')
        if not os.path.exists('./logs'): os.mkdir('./logs')

        nDenseBlocks = (self.nDenseBlock - 4) // 3
        if self.Bottleneck:
            nDenseBlocks //= 2
        # initialization

        population = self.create_init_pop(nDenseBlocks)
        graph = np.array(population)
        np.save('train_data' + str(utils.generation) + "_" + str(utils.prob) + "_" + str(utils.dataset) +'.npy', graph)
        # for row in population:
        #     print(row)
        idx = []
        for p in range(self.pop_size * 3):
            idx.append(self.chk(nDenseBlocks, population[p]))
        # print("idx = ", idx)
        trainloader, testloader, classes = dataloader.dataloader()
        acc = []
        # params = []
        for i in range(1, self.pop_size + 1):
            net = DenseNet(growthRate=utils.growthRate, depth=utils.depth, reduction=utils.reduction,
                           bottleneck=utils.bottleneck, nClasses=utils.nClasses,
                           matrix=population[(i - 1) * self.number_blocks:i * self.number_blocks],
                           idx=idx[(i - 1) * self.number_blocks:i * self.number_blocks]).to(
                device=utils.device)

            # d = torch.empty(64,3,32,32, dtype=torch.float32).to(utils.device)
            # torch.onnx.export(net, d, 'initialization.onnx')

            net = train(net, trainloader, utils.GA_epoch, utils.device)
            accuracy = test(net, testloader, utils.device)

            acc.append(accuracy)
            # params.append(sum(p.numel() for p in net.parameters() if p.requires_grad))

        utils.print_and_log(logger, "prob = {} acc = {}".format(self.prob, acc))
        # utils.print_and_log(logger, "params = {}".format(params))
        fitness = acc  # self.fitness(acc, params)
        # utils.print_and_log(logger, "prob = {} fitness = {}".format(self.prob, fitness))

        surrogate_data = np.load('./train_data' + str(utils.generation) + "_" + str(utils.prob) + '.npy')
        dataset = []
        for i in range(utils.number_population):
            dataset.append(surrogate_data[i*self.number_blocks:(i+1)*self.number_blocks].ravel())
        surrogate_data_df = pd.DataFrame(dataset)
        surrogate_label = pd.DataFrame(acc)
        surrogate_trainset = pd.concat([surrogate_data_df, surrogate_label], axis=1)
        X, y = surrogate_trainset.iloc[:, :-1], surrogate_trainset.iloc[:, -1]
        predictor = GradientBoostingRegressor()
        start = time.process_time()
        predictor.fit(X, y)
        utils.print_and_log(logger, "Time taken by surrogate to train the model {}".format(time.process_time() - start))


        new_population = population
        for generation in range(self.generations):
            utils.print_and_log(logger, "-----" + str(generation + 1) + " generation------")
            selected_number = list(range(self.pop_size))
            for n in range(self.pop_size // 2):
                offspring1 = []
                offspring2 = []
                parents, selected_number = self.selection(selected_number)
                crossover_rand = random.random()
                if crossover_rand <= utils.crossover_rate:
                    offspring1, offspring2 = self.crossover(parents[0], parents[1], new_population, nDenseBlocks)
                    mutate_rand = random.random()
                    if mutate_rand <= utils.mutation_rate:
                        offspring1, offspring2 = self.mutation(offspring1, offspring2, nDenseBlocks)
                offspring1 = self.chk_diagonal(nDenseBlocks, offspring1)
                offspring2 = self.chk_diagonal(nDenseBlocks, offspring2)
                new_population.extend(offspring1)
                new_population.extend(offspring2)
            for p in range(self.pop_size + 1, len(new_population)):
                idx.append(self.chk(nDenseBlocks, new_population[p]))
            # print("idx = ", idx)

            do_surrogate = self.pop_size + int(self.pop_size*0.9)

            for m in range(self.pop_size, self.pop_size * 2):
                if m < do_surrogate:
                    net = DenseNet(growthRate=utils.growthRate, depth=utils.depth, reduction=utils.reduction,
                                   bottleneck=True, nClasses=utils.nClasses,
                                   matrix=new_population[(m - 1) * self.number_blocks:m * self.number_blocks],
                                   idx=idx[(m - 1) * self.number_blocks:m * self.number_blocks]).to(
                        device=utils.device)
                    matrix = np.array(new_population[(m - 1) * self.number_blocks:m * self.number_blocks])
                    test_matrix = matrix.ravel().reshape(1, -1)
                    accuracy = predictor.predict(test_matrix)
                    acc.append(accuracy[0])

                else:
                    net = DenseNet(growthRate=utils.growthRate, depth=utils.depth, reduction=utils.reduction,
                                   bottleneck=True, nClasses=utils.nClasses,
                                   matrix=new_population[(m - 1) * self.number_blocks:m * self.number_blocks],
                                   idx=idx[(m - 1) * self.number_blocks:m * self.number_blocks]).to(
                        device=utils.device)
                    net = train(net, trainloader, utils.GA_epoch, utils.device)
                    accuracy = test(net, testloader, utils.device)
                    acc.append(accuracy)

                    matrix = np.array(new_population[(m - 1) * self.number_blocks:m * self.number_blocks])
                    new_train_matrix = matrix.ravel()
                    dataset.append(new_train_matrix)

            new_surrogate_data = pd.DataFrame(dataset)
            new_label = pd.DataFrame(acc[-4:])
            surrogate_label = pd.concat([surrogate_label, new_label], axis=0, ignore_index=True)
            new_surrogate_trainset = pd.concat([new_surrogate_data, surrogate_label], axis=1)
            X, y = new_surrogate_trainset.iloc[:, :-1], new_surrogate_trainset.iloc[:, -1]
            # predictor = CatBoostRegressor(verbose=0)
            start = time.process_time()
            predictor.fit(X, y)
            utils.print_and_log(logger, "Time taken by surrogate to train the model {}".format(time.process_time() - start))

            utils.print_and_log(logger, "prob = {} acc = {} ".format(self.prob, acc))
            # utils.print_and_log(logger, "params : {}".format(params))
            fitness = acc  # self.fitness(acc, params)

            parents_population = new_population[:self.pop_size * self.number_blocks]
            parents_fitness = fitness[:self.pop_size]
            idx_parents = idx[:self.pop_size * self.number_blocks]
            parents_acc = acc[:self.pop_size]
            # parents_params = params[:self.pop_size]

            offspring_population = new_population[self.pop_size * self.number_blocks:]
            offspring_fitness = fitness[self.pop_size:]
            idx_offspring = idx[self.pop_size * self.number_blocks:]
            offspring_acc = acc[self.pop_size:]
            # offspring_params = params[self.pop_size:]

            parent_rank = np.argsort(parents_fitness)[::-1]
            parents_population_rank = []
            idx_parents_rank = []
            for i in parent_rank + 1:
                parents_population_rank.extend(parents_population[(i - 1) * self.number_blocks:i * self.number_blocks])
                idx_parents_rank.extend(idx_parents[(i - 1) * self.number_blocks:i * self.number_blocks])
            parents_fitness = [parents_fitness[i] for i in parent_rank]
            parents_acc = [parents_acc[i] for i in parent_rank]
            # parents_params = [parents_params[i] for i in parent_rank]

            offspring_rank = np.argsort(offspring_fitness)[::-1]
            offspring_population_rank = []
            idx_offspring_rank = []
            for i in parent_rank + 1:
                offspring_population_rank.extend(
                    offspring_population[(i - 1) * self.number_blocks:i * self.number_blocks])
                idx_offspring_rank.extend(idx_offspring[(i - 1) * self.number_blocks:i * self.number_blocks])
            offspring_fitness = [offspring_fitness[i] for i in offspring_rank]
            offspring_acc = [offspring_acc[i] for i in offspring_rank]
            # offspring_params = [offspring_params[i] for i in offspring_rank]

            elite_rate = utils.elitism
            parents_population = parents_population[:int(self.pop_size * elite_rate) * self.number_blocks]
            parents_fitness = parents_fitness[:int(self.pop_size * elite_rate)]
            idx_parents = idx_parents[:int(self.pop_size * elite_rate) * self.number_blocks]
            parents_acc = parents_acc[:int(self.pop_size * elite_rate)]
            # parents_params = parents_params[:int(self.pop_size*elite_rate)]

            offspring_population = offspring_population[:int(self.pop_size * (1 - elite_rate)) * self.number_blocks]
            offspring_fitness = offspring_fitness[:int(self.pop_size * (1 - elite_rate))]
            idx_offspring = idx_offspring[:int(self.pop_size * (1 - elite_rate)) * self.number_blocks]
            offspring_acc = offspring_acc[:int(self.pop_size * (1 - elite_rate))]
            # offspring_params = offspring_params[:int(self.pop_size*elite_rate)]

            new_population = parents_population + offspring_population  # np.concatenate((parents_population, offspring_population), axis=0)
            fitness = parents_fitness + offspring_fitness
            idx = idx_parents + idx_offspring
            acc = parents_acc + offspring_acc
            # params = parents_params + offspring_params

            rank = np.argsort(fitness)[::-1]
            utils.print_and_log(logger, "Fitness values = {}".format(fitness))
            fitness = [fitness[i] for i in rank]

            # utils.print_and_log(logger, "Fitness values = {}".format(fitness))
            utils.print_and_log(logger, "Best Fitness value = {}".format(fitness[0]))

        rank = np.argsort(fitness)[::-1]
        population_rank = []
        idx_rank = []
        for i in rank + 1:
            population_rank.extend(new_population[(i - 1) * self.number_blocks:i * self.number_blocks])
            idx_rank.extend(idx[(i - 1) * self.number_blocks:i * self.number_blocks])

        idx = idx[:self.number_blocks]
        best_model = new_population[:self.number_blocks]

        net = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10,
                       matrix=best_model[0:self.number_blocks],
                       idx=idx[0:self.number_blocks]).to(
            device=utils.device)
        print(net)
        torch.save(net, './models/model{:%Y%m%d}_{}_{}.pt'.format(datetime.datetime.now(), self.prob,
                                                                  str(utils.augmentation)))
        utils.print_and_log(logger, "# Params = {}".format(sum(p.numel() for p in net.parameters() if p.requires_grad)))
        utils.print_and_log(logger, "Finish")
