# Copyright (c) Mathurin Videau. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from functools import partial
import heapq
from operator import attrgetter, eq
import random
from bisect import bisect_right

from deap.base import Fitness
from deap import tools
import numpy as np

class UCBFitness(Fitness):#calculation of the fitness using UCB
    
    c=np.sqrt(2)
    sigma=1.0

    def __init__(self, offset=0, values=(), **kwargs):
        super().__init__(values=values, **kwargs)
        self.rewards = []
        self.offset = offset

    def add_reward(self, reward):
        self.rewards.append(reward)
    
    def update_offset(self):
        self.offset = len(self.rewards) 

    def calc_fitness(self, budget):
        if self.rewards:
            return np.mean(self.rewards, axis=0) + self.sigma*self.c*np.sqrt(np.log(budget+self.offset+1)/len(self.rewards))
    
    def reset(self):
        self.rewards = []
        self.offset = 0

class HeapWithKey(list):# For speeder arm selection use heap representations
    def __init__(self, initial=[], key=lambda x:x):
       super(HeapWithKey, self).__init__([(key(item), i, item) for i, item in enumerate(initial)])
       self.key = key
       self.idx = 0
       if initial:
           self.idx = len(self)
           heapq.heapify(self)
       else:
           self = []

    def push(self, item):
       heapq.heappush(self, (self.key(item), self.idx, item))
       self.idx += 1

    def pop(self):
       return heapq.heappop(self)[2]

#Selection with double tournament to keep not only best individuals but also most pulled arms 
def selDoubleTournament(individuals, k, fitness_size, parsimony_size, fitness_first=True, fit_attr="fitness"):
    assert (0.0 < parsimony_size <= 2), "Parsimony tournament size has to be in the range [1, 2]."

    def _sizeTournament(individuals, k, select):
        chosen = []
        for i in range(k):
            prob = parsimony_size / 2.
            ind1, ind2 = select(individuals, k=2)

            if len(ind1.fitness.rewards) < len(ind2.fitness.rewards):
                ind1, ind2 = ind2, ind1
            elif len(ind1) == len(ind2):
                prob = 0.5

            chosen.append(ind1 if random.random() < prob else ind2)

        return chosen

    def _fitTournament(individuals, k, select):
        chosen = []
        for i in range(k):
            aspirants = select(individuals, k=fitness_size)
            chosen.append(max(aspirants, key=attrgetter(fit_attr)))
        return chosen

    if fitness_first:
        tfit = partial(_fitTournament, select=tools.selRandom)
        return _sizeTournament(individuals, k, tfit)
    else:
        tsize = partial(_sizeTournament, select=tools.selRandom)
        return _fitTournament(individuals, k, tsize)


class ArmHof(tools.HallOfFame):#Hall of fame of most tested individual
    def __init__(self, maxsize):
        super().__init__(maxsize, similar=eq)
    
    def update(self, population):
        for ind in population:
            for i in range(len(self)):#to avoid duplicate
                    if self.similar(self[i], ind):
                        self.remove(i)
                        break
            if len(self) == 0 and self.maxsize !=0:
                # Working on an empty hall of fame is problematic for the
                # "for else"
                self.insert(population[0])
                continue
            if len(ind.fitness.rewards) > len(self[-1].fitness.rewards) or len(self) < self.maxsize:
                if len(self) >= self.maxsize:
                    self.remove(-1)
                self.insert(ind)
    
    def insert(self, item):
        _item = deepcopy(item)
        _item.fitness.rewards = deepcopy(item.fitness.rewards)
        i = bisect_right(self.keys, len(_item.fitness.rewards))
        self.items.insert(len(self) - i, item)
        self.keys.insert(i, len(_item.fitness.rewards))
    

class UpdateFitnessHof(tools.HallOfFame):#Fitness hof that take into account fitness change of individuals
    def __init__(self, maxsize, similar=eq, maxsize_arm=None):
        super().__init__(maxsize, similar=similar)
        self.maxsize_arm = maxsize_arm
        if maxsize_arm is not None:
            self.arm_hof = ArmHof(self.maxsize_arm)

    def update(self, population):
        for ind in population:
            idx = 0
            while idx < len(self):#to avoid duplicate
                if self.similar(ind, self[idx]):
                    self.remove(idx)
                    break
                idx+=1
            if len(self) == 0 and self.maxsize !=0:
                # Working on an empty hall of fame is problematic for the
                # "for else"
                self.insert(ind)
                continue

            if ind.fitness > self[-1].fitness or len(self) < self.maxsize:
                # The individual is unique and strictly better than
                # the worst
                if len(self) >= self.maxsize:
                    self.remove(-1)
                self.insert(ind)
        if self.maxsize_arm is not None:
            self.arm_hof.update(population)

class UpdateFitnessParetoFront(tools.ParetoFront):#Pareto front that take into account fitness change of individuals
    def __init__(self, similar=eq, maxsize_arm=None):
        super().__init__(similar=similar)
        self.maxsize_arm = maxsize_arm
        if maxsize_arm:
            self.arm_hof = ArmHof(self.maxsize_arm)

    def update(self, population):
        for ind in population:#remove duplicate to update there value
            idx = 0
            while idx < len(self):
                if self.similar(ind, self[idx]):
                    self.remove(idx)
                    idx-=1
                idx+=1

        super().update(population)
        
        if self.maxsize_arm is not None:
            self.arm_hof.update(population)