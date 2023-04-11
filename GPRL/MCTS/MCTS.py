# Copyright (c) Mathurin Videau. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ..genetic_programming.linearGP import Instruction
import numpy as np

import copy
import random

from collections import deque
from itertools import product

class TreeNMCS(object):#Nested monte-carlo to optimise a typed tree of operation
    def __init__(self, pset, max_len, fitness, nb_playout=1):
        self.max_len = max_len
        self.pset = pset
        self.ops = copy.deepcopy(pset.terminals)
        for k, v in pset.primitives.items():
            if k in self.ops:
                self.ops[k] += copy.deepcopy(v)
            else:
                self.ops[k] = copy.deepcopy(v)
        self.nb_playout = nb_playout
        self.fitness = fitness
    
    def possible_primitives(self, individual, leaves):
        l = []
        for op in self.ops[leaves[0]]:
            if isinstance(op, type):
                op = op()
            if len(individual) + len(leaves) + op.arity <= self.max_len:
                l.append(op)
        return l
    
    def play(self, individual, leaves, op):
        individual.append(op)
        leaves.pop(0)
        if op.arity > 0:
            leaves = op.args + leaves
        return individual, leaves
    
    def playout(self, individual, leaves):
        while len(leaves) != 0:
            op = random.choice(self.possible_primitives(individual, leaves))
            individual, leaves = self.play(individual, leaves, op)
        return individual
    
    def compute_fitness(self, individual):
        #f = gp.compile(gp.PrimitiveTree(individual), pset=self.pset)
        return self.fitness(individual)
    
    def run(self, individual, leaves, level):
        best_individual = None
        best_fitness = np.NINF
        while len(leaves) != 0:
            ops = self.possible_primitives(individual, leaves)
            for op in ops:
                cp_ind, cp_leaves = copy.deepcopy(individual), copy.deepcopy(leaves)
                cp_ind, cp_leaves = self.play(cp_ind, cp_leaves, op)
                if level == 1:
                    cp_ind = self.playout(cp_ind, cp_leaves)
                else:
                    cp_ind = self.run(cp_ind, cp_leaves, level-1)
                fitness = self.compute_fitness(cp_ind)

                if fitness > best_fitness:
                    best_individual = copy.deepcopy(cp_ind)
                    best_fitness = fitness
            individual, leaves = self.play(individual, leaves, best_individual[len(individual)])
        return individual

class LinearNCMS(object):#/!\ never been tested used the linear gp representation too build expressions with NMCS
    def __init__(self, interpreter, regCalcSize, regSize, max_len, fitness, nb_playout=1):
        self.max_len = max_len
        self.interpreter = interpreter
        self.nb_playout = nb_playout
        self.fitness = fitness

        self.regCalcSize = regCalcSize
        self.regSize = regSize
    
    def play(self, individual, instruction, R_eff):#add an instruction to the list
        if not self.interpreter.branch[instruction.opcode]: R_eff.remove(instruction.dst)
        
        R_eff.add(instruction.inpt1)
        if self.interpreter.arity[instruction.opcode] == 2:
            R_eff.add(instruction.inpt2)
        
        individual.appendleft(instruction)
        return individual
    
    def possible_primitives(self, individual, R_eff, opsOnly=False):#possible primitives, opsOnly=False -> test also register values, opsOnly=True -> test only ops with random registers
        instructions = []
        out = R_eff.intersection(set(range(self.regCalcsize)))
        if not bool(out):
            out = range(self.regCalcsize)
        inpt = range(self.regSize) if len(individual) < self.max_size else range(self.regCalcSize, self.regSize)
        for opcode, dst, in1, in2 in product(self.interpreter.ops, R_eff, inpt, inpt):
            instructions.append(Instruction(opcode, dst, in1, in2))
        return instructions

    def playout(self, individual, R_eff):
        while len(individual) < self.max_len:
            instr = random.choice(self.possible_primitives(individual, R_eff))
            individual = self.play(individual, instr)
        return individual

    def compute_fitness(self, individual):
        return self.fitness(individual)

    def run(self, individual, level):
        best_individual = None
        best_fitness = np.NINF
        while len(individual) < self.max_len:
            instructions = self.possible_primitives(individual)
            for instr in instructions:
                cp_ind = copy.deepcopy(individual)
                cp_ind = self.play(cp_ind, instr)
                if level == 1:
                    cp_ind = self.playout(cp_ind)
                else:
                    cp_ind = self.run(cp_ind, level-1)
                fitness = self.compute_fitness(cp_ind)

                if fitness > best_fitness:
                    best_individual = copy.deepcopy(cp_ind)
                    best_fitness = fitness
            individual, leaves = self.play(individual, leaves, best_individual[len(individual)])
        return individual