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
from utils import logger, arg
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
        self.pop_size = arg.number_population
        self.generations = arg.generation
        self.nDenseBlock = nDenseBlock
        self.Bottleneck = Bottleneck
        # self.prob = utils.prob
        self.number_blocks = 3

        utils.print_and_log(logger,
                            "GA start generation : " + str(self.generations) + " BC = " + str(arg.bottleneck) + " population : " + str(self.pop_size))

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

        one_prob = [0.9, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
        for i in range(nb_layers):
            for j in range(i + 2, nb_layers):
                prob = random.random()
                # print(prob)
                if prob <= random.choice(one_prob):
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
        ran_num = np.random.rand()
        if ran_num <= 0.33:
            # one-point crossover row
            parent1 = np.array(new_population[p1 * self.number_blocks:(p1 + 1) * self.number_blocks])
            parent2 = np.array(new_population[p2 * self.number_blocks:(p2 + 1) * self.number_blocks])

            point = np.random.randint(1, nDenseBlocks)
            offspring1 = np.concatenate((parent1[:, :point], parent2[:, point:]),
                                        axis=1)
            offspring2 = np.concatenate((parent2[:, :point], parent1[:, point:]),
                                        axis=1)

        elif 0.33 < ran_num and ran_num < 0.66:
            # one-point crossover col
            parent1 = np.array(new_population[p1 * self.number_blocks:(p1 + 1) * self.number_blocks])
            parent2 = np.array(new_population[p2 * self.number_blocks:(p2 + 1) * self.number_blocks])

            point = np.random.randint(1, nDenseBlocks)
            offspring1 = np.concatenate((parent1[:, :point, :], parent2[:, point:, :]),
                                        axis=1)
            offspring2 = np.concatenate((parent2[:, :point, :], parent1[:, point:, :]),
                                        axis=1)

        else:
            parent1 = np.array(new_population[p1 * self.number_blocks:(p1 + 1) * self.number_blocks])
            parent2 = np.array(new_population[p2 * self.number_blocks:(p2 + 1) * self.number_blocks])

            row_point = np.random.randint(1, nDenseBlocks)
            col_point = np.random.randint(1, nDenseBlocks)

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

        offspring1 = offspring1.tolist()
        offspring2 = offspring2.tolist()
        return offspring1, offspring2

    def mutation(self, offspring1, offspring2, nDenseBlocks):
        ran_num = np.random.rand()
        if ran_num <= 0.33:
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

        elif 0.33 < ran_num and ran_num < 0.66:
            # bit-flipping row
            off1 = np.array(offspring1)
            off2 = np.array(offspring2)

            row_index = np.random.randint(0, nDenseBlocks+1)
            for i in range(self.number_blocks):
                copy_off1 = off1[i][row_index][:]
                copy_off2 = off2[i][row_index][:]
                for j in range(row_index+1, nDenseBlocks):
                    copy_off1[j] = 1 - off1[i][row_index][j]
                    copy_off2[j] = 1 - off2[i][row_index][j]
                off1[i][row_index][:] = copy_off1[:]
                off2[i][row_index][:] = copy_off2[:]
        else:
            # bit-flipping col
            off1 = np.array(offspring1)
            off2 = np.array(offspring2)

            col_index = np.random.randint(2, nDenseBlocks + 1)
            for i in range(self.number_blocks):
                copy_off1 = off1[i][:, col_index]
                copy_off2 = off2[i][:, col_index]
                for j in range(col_index):
                    copy_off1[j] = 1 - off1[i][j, col_index]
                    copy_off2[j] = 1 - off2[i][j, col_index]
                off1[i][:, col_index] = copy_off1[:]
                off2[i][:, col_index] = copy_off2[:]


        offspring1 = off1.tolist()
        offspring2 = off2.tolist()

        return offspring1, offspring2
    def surrogate(self, label, df):
        surrogate_trainset = pd.concat([df, label], axis=1)
        X, y = surrogate_trainset.iloc[:, :-1], surrogate_trainset.iloc[:, -1]
        predictor = DecisionTreeRegressor()
        # predictor = GradientBoostingRegressor()
        start = time.process_time()
        predictor.fit(X, y)
        utils.print_and_log(logger, "Time taken by surrogate to train the model {}".format(time.process_time() - start))
        return predictor

    def to_MB(self, a):
        return a / 1024.0 / 1024.0
    def evolve(self):
        if not os.path.exists('./models'): os.mkdir('./models')
        if not os.path.exists('./logs'): os.mkdir('./logs')

        nDenseBlocks = (self.nDenseBlock - 4) // 3
        if self.Bottleneck:
            nDenseBlocks //= 2

        population = self.create_init_pop(nDenseBlocks)

        idx = []
        for p in range(self.pop_size * 3):
            idx.append(self.chk(nDenseBlocks, population[p]))

        trainloader, testloader, classes = dataloader.GAdataloader()
        acc = []
        dataset = []
        train_rate = 0.1
        for i in range(1, self.pop_size):
            net = DenseNet(growthRate=arg.growthRate, depth=arg.depth, reduction=arg.reduction,
                           bottleneck=arg.bottleneck, nClasses=arg.nClasses,
                           matrix=population[(i - 1) * self.number_blocks:i * self.number_blocks],
                           idx=idx[(i - 1) * self.number_blocks:i * self.number_blocks]).to(
                device=arg.device)


            # print(f"After model to device: {self.to_MB(torch.cuda.memory_allocated()):.2f}MB")
            torch.cuda.empty_cache()
            net, loss = GAtrain(net, trainloader, arg.GA_epoch, arg.device)
            acc.append(loss)
            matrix = np.array(population[(i - 1) * self.number_blocks:i * self.number_blocks])
            new_train_matrix = matrix.ravel()
            dataset.append(new_train_matrix)
        if arg.surrogate == "True":
            surrogate_data_df = pd.DataFrame(dataset)
            surrogate_label = pd.DataFrame(acc)
            predictor = self.surrogate(surrogate_label, surrogate_data_df)


        # for n in range(int(self.pop_size * train_rate), self.pop_size):
        #     matrix = np.array(population[(n - 1) * self.number_blocks:n * self.number_blocks])
        #     test_matrix = matrix.ravel().reshape(1, -1)
        #     accuracy = predictor.predict(test_matrix)
        #     acc.append(accuracy[0])
        utils.print_and_log(logger, "loss = {}".format(acc))
        fitness = acc


        new_population = population
        for generation in range(self.generations):
            utils.print_and_log(logger, "-----" + str(generation + 1) + " generation------")
            selected_number = list(range(len(new_population)//3))
            for n in range(len(new_population)//3//2):
                seed = np.random.rand()
                parents, selected_number = self.selection(selected_number, seed)
                offspring1, offspring2 = self.crossover(parents[0], parents[1], new_population, nDenseBlocks)
                mutate_rand = np.random.rand()
                if mutate_rand <= arg.mutation_rate:
                    offspring1, offspring2 = self.mutation(offspring1, offspring2, nDenseBlocks)
                offspring1 = self.chk_diagonal(nDenseBlocks, offspring1)
                offspring2 = self.chk_diagonal(nDenseBlocks, offspring2)
                new_population.extend(offspring1)
                new_population.extend(offspring2)

            for p in range(self.pop_size + 1, len(new_population)):
                idx.append(self.chk(nDenseBlocks, new_population[p]))

            do_GAtrain = int(self.pop_size*train_rate)
            if arg.surrogate == "True":
                for m in range(self.pop_size, len(new_population)//3):
                    matrix = np.array(new_population[(m - 1) * self.number_blocks:m * self.number_blocks])
                    test_matrix = matrix.ravel().reshape(1, -1)
                    accuracy = predictor.predict(test_matrix)
                    acc.append(accuracy[0])
                chd_surrogate_acc = acc[self.pop_size:]
                top4_chd = sorted(list(enumerate(chd_surrogate_acc)), key=lambda x: x[1])
                top4_chd = [index for index, value in top4_chd[:do_GAtrain]]
                add_label = []
                for do in range(do_GAtrain):
                    net = DenseNet(growthRate=arg.growthRate, depth=arg.depth, reduction=arg.reduction,
                                   bottleneck=arg.bottleneck, nClasses=arg.nClasses,
                                   matrix=new_population[((self.pop_size+top4_chd[do]) - 1) * self.number_blocks:(self.pop_size+top4_chd[do]) * self.number_blocks],
                                   idx=idx[((self.pop_size+top4_chd[do]) - 1) * self.number_blocks:(self.pop_size+top4_chd[do]) * self.number_blocks]).to(
                        device=arg.device)
                    torch.cuda.empty_cache()
                    net, loss = GAtrain(net, trainloader, arg.GA_epoch, arg.device)
                    acc[self.pop_size+top4_chd[do]] = loss
                    add_label.append(loss)

                    matrix = np.array(new_population[((self.pop_size+top4_chd[do]) - 1) * self.number_blocks:(self.pop_size+top4_chd[do]) * self.number_blocks])
                    new_train_matrix = matrix.ravel()
                    dataset.append(new_train_matrix)

                surrogate_data_df = pd.DataFrame(dataset)
                new_label = pd.DataFrame(add_label)
                surrogate_label = pd.concat([surrogate_label, new_label], axis=0, ignore_index=True)
                predictor = self.surrogate(surrogate_label, surrogate_data_df)
            else:
                print("daasf")
                for m in range(self.pop_size, len(new_population)//3):
                    net = DenseNet(growthRate=arg.growthRate, depth=arg.depth, reduction=arg.reduction,
                                   bottleneck=arg.bottleneck, nClasses=arg.nClasses,
                                   matrix=population[(m - 1) * self.number_blocks:m * self.number_blocks],
                                   idx=idx[(m - 1) * self.number_blocks:m * self.number_blocks]).to(
                        device=arg.device)


                    # print(f"After model to device: {self.to_MB(torch.cuda.memory_allocated()):.2f}MB")
                    torch.cuda.empty_cache()
                    net, loss = GAtrain(net, trainloader, arg.GA_epoch, arg.device)
                    acc.append(loss)


            utils.print_and_log(logger, "loss = {}".format(acc))
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

            elite_rate = arg.elitism
            parents_population = parents_population_rank[:int(self.pop_size * elite_rate) * self.number_blocks]
            parents_fitness = parents_fitness[:int(self.pop_size * elite_rate)]
            # parents_second_fitness = parents_second_fitness[:int(self.pop_size * elite_rate)]
            idx_parents = idx_parents_rank[:int(self.pop_size * elite_rate) * self.number_blocks]
            parents_acc = parents_acc[:int(self.pop_size * elite_rate)]
            # parents_params = parents_params[:int(self.pop_size*elite_rate)]

            offspring_population = offspring_population_rank[:int(self.pop_size * (1 - elite_rate)) * self.number_blocks]
            offspring_fitness = offspring_fitness[:int(self.pop_size * (1 - elite_rate))]
            # offspring_second_fitness = offspring_second_fitness[:int(self.pop_size * (1 - elite_rate))]
            idx_offspring = idx_offspring_rank[:int(self.pop_size * (1 - elite_rate)) * self.number_blocks]
            offspring_acc = offspring_acc[:int(self.pop_size * (1 - elite_rate))]

            # offspring_params = offspring_params[:int(self.pop_size*elite_rate)]
            new_population = parents_population + offspring_population  # np.concatenate((parents_population, offspring_population), axis=0)
            fitness = parents_fitness + offspring_fitness
            # second_fitness = parents_second_fitness + offspring_second_fitness
            idx = idx_parents + idx_offspring
            acc = parents_acc + offspring_acc
            # params = parents_params + offspring_params

            rank = np.argsort(fitness)
            utils.print_and_log(logger, "Fitness values = {}".format(fitness))
            best_fitness = [fitness[i] for i in rank]
            utils.print_and_log(logger, "Best Fitness value = {}".format(best_fitness[0]))

        rank = np.argsort(fitness)
        population_rank = []
        idx_rank = []
        for i in rank + 1:
            population_rank.extend(new_population[(i - 1) * self.number_blocks:i * self.number_blocks])
            idx_rank.extend(idx[(i - 1) * self.number_blocks:i * self.number_blocks])

        idx = idx_rank[:self.number_blocks]
        best_model = population_rank[:self.number_blocks]

        net = DenseNet(growthRate=arg.growthRate, depth=arg.depth, reduction=arg.reduction,
                               bottleneck=arg.bottleneck, nClasses=arg.nClasses,
                       matrix=best_model[0:self.number_blocks],
                       idx=idx[0:self.number_blocks]).to(
            device=arg.device)
        print(net)
        torch.save(net, './models/model{:%Y%m%d}_{}.pt'.format(datetime.datetime.now(),
                                                                  str(arg.augmentation)))
        utils.print_and_log(logger, "# Params = {}".format(sum(p.numel() for p in net.parameters() if p.requires_grad)))
        utils.print_and_log(logger, "Finish")

