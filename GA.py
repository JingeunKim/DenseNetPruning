import random
import model
from train import train
from test import test
import utils
import dataloader
import numpy as np
import torch
import os
import datetime
from utils import logger


class GA():
    def __init__(self, nDenseBlock, Bottleneck):
        self.pop_size = utils.number_population
        self.generations = utils.generation
        self.nDenseBlock = nDenseBlock
        self.Bottleneck = Bottleneck

        self.number_blocks = 3

        utils.print_and_log(logger,
                            "GA start generation : " + str(self.generations) + " population : " + str(self.pop_size))

    def create_init_pop(self, nb_layers):
        population = []
        for p in range(self.pop_size):
            for p in range(self.number_blocks):
                population.append(self.create_matrix(nb_layers))
        return population

    def create_matrix(self, nb_layers):
        nb_layers = nb_layers + 1
        # 2차원 배열 초기화
        matrix = [[0 for _ in range(nb_layers)] for _ in range(nb_layers)]

        # 대각선 위 요소를 1로 채우기
        for i in range(nb_layers - 1):
            matrix[i][i + 1] = 1

        # 대각선 위 요소 이외의 요소를 랜덤하게 0 또는 1로 채우기
        for i in range(nb_layers):
            for j in range(i + 2, nb_layers):
                matrix[i][j] = random.choice([0, 1])
        # for row in matrix:
        #     print(row)
        # print()
        return matrix

    def chk(self, nb_layers, matrix):
        idx = []
        for j in range(nb_layers):
            z = [i[j] for i in matrix]
            find_one = []
            for a in range(len(z)):
                if z[a] == 1:
                    find_one.append(a)
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
        # one-point crossover
        parent1 = np.array(new_population[p1 * self.number_blocks:(p1 + 1) * self.number_blocks])
        parent2 = np.array(new_population[p2 * self.number_blocks:(p2 + 1) * self.number_blocks])

        point = random.randint(1, nDenseBlocks)

        offspring1 = np.concatenate((parent1[:, :point], parent2[:, point:]),
                                    axis=1)
        offspring2 = np.concatenate((parent2[:, :point], parent1[:, point:]),
                                    axis=1)

        offspring1 = offspring1.tolist()
        offspring2 = offspring2.tolist()

        return offspring1, offspring2

    def mutation(self, offspring1, offspring2, nDenseBlocks):
        # random point bit-flip mutation
        off1 = np.array(offspring1)
        off2 = np.array(offspring2)

        row_index = random.randint(0, nDenseBlocks - 3)

        col_index = random.randint(row_index + 1, row_index + 2)

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
        if not os.path.exists('./models'): os.mkdir('./models')
        if not os.path.exists('./logs'): os.mkdir('./logs')

        nDenseBlocks = (self.nDenseBlock - 4) // 3
        if self.Bottleneck:
            nDenseBlocks //= 2
        # initialization

        population = self.create_init_pop(nDenseBlocks)
        # print(population)
        idx = []
        for p in range(self.pop_size * 3):
            idx.append(self.chk(nDenseBlocks, population[p]))
        trainloader, testloader, classes = dataloader.dataloader()
        acc = []
        params = []
        for i in range(1, self.pop_size + 1):
            net = model.DenseNet(growthRate=utils.growthRate, depth=utils.depth, reduction=utils.reduction,
                                 bottleneck=utils.bottleneck, nClasses=utils.nClasses,
                                 matrix=population[(i - 1) * self.number_blocks:i * self.number_blocks],
                                 idx=idx[(i - 1) * self.number_blocks:i * self.number_blocks]).to(
                device=utils.device)
            net = train(net, trainloader, utils.GA_epoch, utils.device)
            accuracy = test(net, testloader, utils.device)
            acc.append(accuracy)
            params.append(sum(p.numel() for p in net.parameters() if p.requires_grad))

        print("acc = ", acc)
        print("params = ", params)
        fitness = self.fitness(acc, params)
        print(fitness)

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
                new_population.extend(offspring1)
                new_population.extend(offspring2)
            for p in range(self.pop_size + 1, len(new_population)):
                idx.append(self.chk(nDenseBlocks, new_population[p]))
            # for row in new_population:
            #     print(row)
            # print()
            for m in range(self.pop_size, self.pop_size * 2):
                net = model.DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10,
                                     matrix=new_population[(m - 1) * self.number_blocks:m * self.number_blocks],
                                     idx=idx[(m - 1) * self.number_blocks:m * self.number_blocks]).to(
                    device=utils.device)
                net = train(net, trainloader, utils.GA_epoch, utils.device)
                accuracy = test(net, testloader, utils.device)
                acc.append(accuracy)
                params.append(sum(p.numel() for p in net.parameters() if p.requires_grad))
                # print(accuracy)

            print("params : {}".format(params))
            fitness = self.fitness(acc, params)
            print("fitness : {}".format(fitness))

            rank = np.argsort(fitness)[::-1]
            acc = np.array(acc)[rank]
            params = np.array(params)[rank]
            acc = acc[:self.pop_size]
            params = params[:self.pop_size]
            acc = acc.tolist()
            params = params.tolist()

            parents_population = new_population[:self.pop_size * self.number_blocks]
            parents_fitness = fitness[:self.pop_size]
            idx_parents = idx[:self.pop_size * self.number_blocks]

            offspring_population = new_population[self.pop_size * self.number_blocks:]
            offspring_fitness = fitness[self.pop_size:]
            idx_offspring = idx[self.pop_size * self.number_blocks:]

            parent_rank = np.argsort(parents_fitness)[::-1]
            parents_population_rank = []
            idx_parents_rank = []
            for i in parent_rank + 1:
                parents_population_rank.extend(parents_population[(i - 1) * self.number_blocks:i * self.number_blocks])
                idx_parents_rank.extend(idx_parents[(i - 1) * self.number_blocks:i * self.number_blocks])
            parents_fitness = [parents_fitness[i] for i in parent_rank]
            # idx_parents = [idx_parents[i] for i in parent_rank]

            offspring_rank = np.argsort(offspring_fitness)[::-1]
            offspring_population_rank = []
            idx_offspring_rank = []
            for i in parent_rank + 1:
                offspring_population_rank.extend(
                    offspring_population[(i - 1) * self.number_blocks:i * self.number_blocks])
                idx_offspring_rank.extend(idx_offspring[(i - 1) * self.number_blocks:i * self.number_blocks])
            offspring_fitness = [offspring_fitness[i] for i in offspring_rank]
            # idx_offspring = [idx_offspring[i] for i in offspring_rank]

            elite_rate = 0.2
            # print("upper 80% = ", int(self.pop_size * elite_rate))
            parents_population = parents_population[:int(self.pop_size * elite_rate) * self.number_blocks]
            parents_fitness = parents_fitness[:int(self.pop_size * elite_rate)]
            idx_parents = idx_parents[:int(self.pop_size * elite_rate) * self.number_blocks]

            offspring_population = offspring_population[:int(self.pop_size * (1 - elite_rate)) * self.number_blocks]
            offspring_fitness = offspring_fitness[:int(self.pop_size * (1 - elite_rate))]
            idx_offspring = idx_offspring[:int(self.pop_size * (1 - elite_rate)) * self.number_blocks]

            new_population = parents_population + offspring_population  # np.concatenate((parents_population, offspring_population), axis=0)
            fitness = parents_fitness + offspring_fitness
            idx = idx_parents + idx_offspring
            rank = np.argsort(fitness)[::-1]
            fitness = fitness[rank]
            utils.print_and_log(logger, "Fitness values = {}".format(fitness))
            utils.print_and_log(logger, "Best Fitness value = {}".format(fitness[0]))

        rank = np.argsort(fitness)[::-1]
        population_rank = []
        idx_rank = []
        for i in rank + 1:
            population_rank.extend(new_population[(i - 1) * self.number_blocks:i * self.number_blocks])
            idx_rank.extend(idx[(i - 1) * self.number_blocks:i * self.number_blocks])

        idx = idx[:self.number_blocks]
        best_model = new_population[:self.number_blocks]
        print(idx)
        print(best_model)
        net = model.DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10,
                             matrix=best_model[0:self.number_blocks],
                             idx=idx[0:self.number_blocks]).to(
            device=utils.device)
        print(net)
        torch.save(net, './models/model{:%Y%m%d}.pt'.format(datetime.datetime.now()))
        utils.print_and_log(logger, "# Params = {}".format(sum(p.numel() for p in net.parameters() if p.requires_grad)))
        utils.print_and_log(logger, "Finish")
