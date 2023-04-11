# Copyright (c) Mathurin Videau. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from functools import partial
import copy

from abc import ABC, abstractmethod

import numpy as np

from collections import namedtuple

"""
Implementation based on :
https://github.com/ChengyuanSha/SMILE/blob/master/linear_genetic_programming/_program.py
and
Book : Brameier, M. F., & Banzhaf, W. (2007). Linear genetic programming. Springer Science & Business Media.
"""
NUM_OPS = 12

opcode_complexity = np.array([1, 1, 1, 2, 2, 4, 4, 4, 3, 4, 4, 4])

class Interpreter(ABC):# Abstract class to define custom interpreter

    @abstractmethod
    def opcode(code, x1, x2):
        pass

    @abstractmethod
    def toString(opcode):
        pass


class BasicInterpreter(Interpreter):
    data_type="float"
    def __init__(self, mask=np.ones(NUM_OPS, dtype=bool)):
        self.num_ops = NUM_OPS
        self.arity = np.array([2]*6 + [1]*3 + [2]*3)
        self.branch = np.zeros(self.num_ops, dtype=bool)
        self.branch[10:] = True
        self.masked_branch = self.branch[mask].copy()# for random selection of operator
        self.ops = np.arange(0, self.num_ops)[mask]
    
    def opcode(self, code, x1, x2):#code to operator translation
        c_undef = 1e7
        if code==0:
            return np.add(x1, x2)
        elif code==1:
            return np.subtract(x1, x2)
        elif code==2:
            return np.multiply(x1, x2)
        elif code==3:
            return np.float_power(np.abs(x1),x2, where=np.logical_and(np.abs(x2) <= 10, np.logical_and(np.abs(x1) > 0.001, np.abs(x1) < 1000)), out=np.full_like(x1, c_undef))
        elif code==4:
            return np.divide(x1, x2, where=np.abs(x2) > 0.001, out=np.full_like(x1, c_undef))
        elif code==5:
            return np.fmod(x1, x2, where=np.abs(x2) > 0.001, out=np.zeros_like(x1))
        elif code==6:
            return np.sin(x1)
        elif code==7:
            return np.exp(x1, where=np.abs(x1)<32, out=np.full_like(x1, c_undef))
        elif code==8:
            return np.log(np.abs(x1), where=np.abs(x1) > 0.00001,  out=np.full_like(x1, c_undef))

        elif code==9:
            return np.where(x1 > x2, -x1, x1)
        elif code==10:
            return x1>x2
        elif code==11:
            return x1<x2
        raise ValueError("Code not found")

    def toString(self, x):
        if x==0:
            return "+"
        elif x==1:
            return "-"
        elif x==2:
            return "*"
        elif x==3:
            return "^"
        elif x==4:
            return "/"
        elif x==5:
            return "%"
        elif x==6:
            return "sin"
        elif x==7:
            return "exp"
        elif x==8:
            return "log"
        elif x==9:
            return "if neg"
        elif x==10:
            return ">"
        elif x==11:
            return "<"
        
class id(object):
    id = 0
    def __call__(self):
        self.id +=1
        return self.id

Instruction = namedtuple('Instruction', ['opcode', 'dst', 'inpt1', 'inpt2'])

class Program(list):
    instruction_type = [('opcode', 'u8'), ('dst', 'u8'), ('inpt1','u8'),('inpt2','u8')]
    Instruction = Instruction

    ID = id()

    def __init__(self, content=[], regCalcSize=None, regInputSize=None, regConstSize=None, regConst=None, interpreter=BasicInterpreter()):
        super(Program, self).__init__(content)
        self.regCalcSize = regCalcSize
        self.regInputSize = regInputSize
        self.regConstSize = regConstSize

        self.regConst = regConst

        self.interpreter = interpreter

        self.id = self.ID()

    def to_effective(self, outputIdxs, stopAt=0):
        effective = []
        idxs = []#index of the effective instructions in self (for mutation pupose)
        R_eff = set(outputIdxs)
        k = len(self)-1
        while k >= stopAt:
            instr = self[k]
            if instr.dst in R_eff and not self.interpreter.branch[instr.opcode]:
                effective.insert(0, instr)
                idxs.append(k)
                if k>0 and not self.interpreter.branch[self[k-1].opcode]: R_eff.remove(instr.dst)
                arity = self.interpreter.arity[instr.opcode]
                R_eff.add(instr.inpt1)
                if arity == 2:
                    R_eff.add(instr.inpt2)
                i = k-1
                while i>=0 and self.interpreter.branch[self[i].opcode]:
                    R_eff.add(self[i].inpt1)
                    R_eff.add(self[i].inpt2)
                    effective.insert(0, self[i])
                    idxs.append(i)
                    i-=1
                k = i+1
            k-=1
        return Program(effective, self.regCalcSize, self.regInputSize, self.regConstSize, self.regConst, interpreter=self.interpreter), R_eff, idxs
    
    def get_used_regConstIdxs(self):
        regConstIdx = set()
        for instr in self:
            if instr.inpt1 >= self.regCalcSize+self.regInputSize:
                regConstIdx.add(instr.inpt1)
            elif self.interpreter.arity[instr.opcode] == 2 and instr.inpt2 >= self.regCalcSize+self.regInputSize:
                regConstIdx.add(instr.inpt2)
        return regConstIdx

    def get_constIdxs(self):
        constIdxs = []
        for idx, instr in enumerate(self):
            if instr.inpt1 >= self.regCalcSize+self.regInputSize:
                constIdxs.append((idx, instr.inpt1))
            elif self.interpreter.arity[instr.opcode] == 2 and instr.inpt2 >= self.regCalcSize+self.regInputSize:
                constIdxs.append((idx, instr.inpt2))
        return constIdxs

    def to_numpy(self):
        return np.array(self, dtype=self.instruction_type)

    def __str__(self):
        string = "   op dst inpt1 inpt2\n"
        for line, instr in enumerate(self):
            string+= f"{line} {self.interpreter.toString(instr.opcode)} {instr.dst} {instr.inpt1} {instr.inpt2}\n"
        return string

    def init_register(self, random=lambda nb:np.arange(1, nb+1, dtype="float")):
        register_length = self.regCalcSize + self.regConstSize + self.regInputSize
        register = np.zeros(register_length, dtype="float")
        register[self.regCalcSize:] = 100.0
        register[self.regCalcSize:self.regCalcSize+self.regInputSize] = np.random.uniform(-1,1, self.regInputSize)
        # initialize constant
        j = self.regCalcSize + self.regInputSize
        if self.regConst is not None:
            register[j:] = self.regConst.copy() 
        else:
            register[j:] = random(self.regConstSize)
        return register

    @classmethod
    def randomProgram(cls, regCalcSize, regInputSize, regConstSize, length, pConst, pBranch, random=lambda nb:np.arange(1, nb+1, dtype="float"), ops=np.array([True, True, True, False, True, False, False, False, False, False, True, True])):
        interpreter = BasicInterpreter(mask=ops)
        prgm = [cls.randomInstruction(regCalcSize, regInputSize, regConstSize, pConst, pBranch, interpreter) for _
                    in range(length - 1)]
        prgm.append(cls.randomInstruction(regCalcSize, regInputSize, regConstSize, pConst, 0.0, interpreter))
        return cls(prgm, regCalcSize, regInputSize, regConstSize, random(regConstSize), interpreter)

    @classmethod
    def randomInstruction(cls, numberOfVariable, numberOfInput, numberOfConstant, pConst, pBranch, interpreter):
        if np.random.random() >= pConst:  # reg1 will be a variable or input
            r1 = np.random.randint(numberOfVariable + numberOfInput)
            reg1Index = r1
            if np.random.random() >= pConst:  # reg2 will be a variable or input
                r2 = np.random.randint(numberOfVariable + numberOfInput)
                reg2Index = r2
            else:  # reg2 will be a constant
                r2 = np.random.randint(numberOfConstant)
                reg2Index = numberOfVariable + numberOfInput + r2
        else:  # reg1 will be a constant and reg2 will be a variable or input
            r1 = np.random.randint(numberOfConstant)
            reg1Index = numberOfVariable + numberOfInput + r1
            r2 = np.random.randint(numberOfVariable + numberOfInput)
            reg2Index = r2
        if np.random.random() < pBranch:
            branch_ops = np.random.choice(interpreter.ops[interpreter.masked_branch])
            return cls.Instruction(branch_ops, 255, reg1Index, reg2Index)
        else:
            operIndex = np.random.choice(interpreter.ops[~interpreter.masked_branch])
            # since zero is return register in calculation, make sure there are enough zeros by increasing its chance
            #pZero = 0.0004 * numberOfInput
            #returnRegIndex = 0 if pZero > np.random.random_sample() else np.random.randint(numberOfVariable)
            returnRegIndex = np.random.randint(numberOfVariable)
            return cls.Instruction(operIndex, returnRegIndex, reg1Index, reg2Index)
    
    @staticmethod
    #MicroMutation
    def mutInstr(prog, pReg, pOp, pConst, pBranch, sigma=1.0, effective=None):
        if not prog: return prog
        
        if effective:#only make effective mutation
            eff, _, idxs =prog.to_effective(effective)
            idx = random.choice(idxs) if idxs else random.randint(0, len(prog)-1)
        else:
            idx = random.randint(0, len(prog)-1)

        opcode, dst, inpt1, inpt2 = prog[idx]

        mut = random.choices(range(0,3), weights=[pReg, pOp, pConst])[0]

        if mut == 0:
            if random.random() < 0.5 and not prog.interpreter.branch[prog[idx].opcode]:
                if effective:
                    authorized_dst = set(range(prog.regCalcSize))
                    _, R_eff, _ = prog.to_effective(effective, stopAt=idx)
                    R_eff.intersection_update(authorized_dst)
                    if bool(R_eff):
                        dst = np.random.choice(tuple(R_eff))
                    else:
                        dst = np.random.randint(prog.regCalcSize)
                else:    
                    dst = np.random.randint(prog.regCalcSize)
            else:
                if random.random() >= pConst: 
                    if prog.interpreter.arity[prog[idx].opcode]==1 or random.random() < 0.5:
                        inpt1 = np.random.randint(prog.regCalcSize + prog.regInputSize)
                    else:
                        inpt2 = np.random.randint(prog.regCalcSize + prog.regInputSize)
                else:
                    r = prog.regCalcSize + prog.regInputSize
                    if prog.interpreter.arity[prog[idx].opcode]==1 or random.random() < 0.5:
                        inpt1 = r + np.random.randint(prog.regConstSize)
                    else:
                        inpt2 = r + np.random.randint(prog.regConstSize)
        elif mut == 1:
            if random.random() < pBranch:# attention si ops interpreter aucune branch
                opcode = np.random.choice(prog.interpreter.ops[prog.interpreter.masked_branch])
            else:
                if prog.interpreter.branch[opcode] and dst>prog.regCalcSize: dst = np.random.randint(prog.regCalcSize)
                opcode = np.random.choice(prog.interpreter.ops[~prog.interpreter.masked_branch])
        elif mut == 2:
            if effective:
                constIdxs = eff.get_constIdxs()
            else:
                constIdxs = prog.get_constIdxs()
            if constIdxs:
                _, inpt = random.choice(constIdxs)
                prog.regConst[inpt-(prog.regCalcSize+prog.regInputSize)] += np.random.normal(loc=0.0, scale=sigma)
        
        if mut!=2:
            prog[idx] = prog.Instruction(opcode, dst, inpt1, inpt2)
        
        return prog
    
    @staticmethod
    def execute(program, inputs, register, outputIdxs):
        if inputs.ndim == 1:
            assert inputs.size == program.regInputSize
            register[program.regCalcSize:program.regCalcSize+inputs.shape[0]] = inputs.copy()
            output = Program._execute(program, register)
            return output[outputIdxs]
        elif inputs.ndim == 2:# speed up the programm execution by using numpy array operator
            assert inputs.shape[1] == program.regInputSize
            ndim_register = np.zeros((register.shape[0], inputs.shape[0]))
            for k in range(inputs.shape[0]):
                ndim_register[:program.regCalcSize, k] = register[:program.regCalcSize].copy()
                ndim_register[program.regCalcSize:program.regCalcSize+inputs.shape[1], k] = inputs[k].copy()
                ndim_register[program.regCalcSize+inputs.shape[1]:, k] = register[program.regCalcSize+inputs.shape[1]:].copy()
            output = Program._execute(program, ndim_register)
            return output[outputIdxs,:].T
        raise ValueError("Unsuported inputs dimension")

    @staticmethod
    def _execute(program, register):
        check_float_range = lambda x: np.clip(x , -np.sqrt(np.finfo(program.interpreter.data_type).max), np.sqrt(np.finfo(program.interpreter.data_type).max))
        branch_flag = False
        i = 0
        while i < len(program):
            instr = program[i]
            if program.interpreter.branch[instr.opcode]:  # branch Instruction
                tmp = program.interpreter.opcode(instr.opcode, register[instr.inpt1], register[instr.inpt2])

                if branch_flag:# consecutive if
                    np.logical_and(tmp, mask)
                else:
                    mask = tmp
                
                if ~mask.all():#if no input verify the condition skip subsequent instructions
                    branch_flag = False
                    while i < len(program) - 1 and program.interpreter.branch[program[i + 1].opcode]:  # if next is still a branch
                        i += 1
                    i += 2
                else:  # if branch true, execute next instruction
                    branch_flag = True
                    i += 1
            else:  # not a branch Instruction
                if branch_flag:
                    register[instr.dst] = np.where(mask, check_float_range(program.interpreter.opcode(instr.opcode, register[instr.inpt1], register[instr.inpt2])), register[instr.dst])

                    branch_flag = False
                else:
                    register[instr.dst] = check_float_range(program.interpreter.opcode(instr.opcode, register[instr.inpt1], register[instr.inpt2]))
                i += 1
        return register

def graph(prog, outputIdxs, debug=False, terminals_name=None):
    """
        Gives graph representation of the programm. Use pygraphviz to get the graph (see tutorial.ipynb)
            arguments:
                prog: the programm to draw as a graph
                outputIdx: registers used as output
                debug: Show instruction number in node of the graph (optional)
                terminals_name: names of the terminials nodes to display (optionnal)
            return:
                nodes, edges, labels, branch_edges
    """
    prgm, _, _ = prog.to_effective(outputIdxs)
    start = prog.regCalcSize + prog.regInputSize + prog.regConstSize
    nodes = []
    edges = []
    branch_edges = []
    labels = {}

    nodes_dst = {}
    terminal_nodes = set()

    branch_flag = False
    for k, instr in enumerate(prgm):
        nodes.append(k+start)
        if debug:
            labels[k+start] = str(k)+ "\n" + prog.interpreter.toString(instr.opcode)
        else:
            labels[k+start] = prog.interpreter.toString(instr.opcode)
        
        arity = prog.interpreter.arity[instr.opcode]
        if arity==2:
            if instr.inpt2 in nodes_dst.keys():
                edges.append((nodes_dst[instr.inpt2], k+start))
            else:
                terminal_nodes.add(instr.inpt2)
                edges.append((instr.inpt2, k+start))

        if instr.inpt1 in nodes_dst.keys():
            edges.append((nodes_dst[instr.inpt1], k+start))
        else:
            terminal_nodes.add(instr.inpt1)
            edges.append((instr.inpt1, k+start))

        
        if not prog.interpreter.branch[instr.opcode]:
            if branch_flag and instr.dst in nodes_dst.keys():
                branch_edges.append((nodes_dst[instr.dst], k+start))
            nodes_dst[instr.dst] = k+start
            branch_flag = False
        elif k<len(prgm)-1:
            branch_flag = True
            edges.append((k+start, k+1+start))
        
    for k in outputIdxs:
        if k in nodes_dst.keys():
            labels[nodes_dst[k]] += '\n Out'+str(k)

    for k in terminal_nodes:
        nodes.append(k)
        if terminals_name:
            labels[k] = terminals_name[k]
        elif k < prog.regCalcSize:
            labels[k] = "Calc" + str(k)
        elif k >= prog.regCalcSize and k < prog.regCalcSize+prog.regInputSize:
            labels[k] = "ARG" + str(k-prog.regCalcSize) 
        else:
            labels[k] = 'Const' + str(k-prog.regCalcSize-prog.regInputSize)
        if debug:
            labels[k] += "\n"+str(k) 
    
    return nodes, edges, labels, branch_edges


def edit_distance(p1, p2):
    def to_string(program):
        repr = ""
        for instr in program:
            repr+=str(instr.opcode)+str(instr.dst)+str(instr.inpt1)
            if program.interpreter.arity[instr.opcode] == 2:
                repr+=str(instr.inpt2)
        return repr
    s1, s2 = to_string(p1), to_string(p2)
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]
   
def initProgam(pcls, regCalcSize, regInputSize, regConstSize, pConst, pBranch, min_, max_, rnd=None, ops=None):
    kwargs = dict()
    if rnd is not None:
        kwargs['random'] = rnd
    if ops is not None:
        kwargs['ops'] = ops
    prgm = Program.randomProgram(regCalcSize, regInputSize, regConstSize, random.randint(min_, max_), pConst, pBranch, **kwargs)
    return pcls(prgm, prgm.regCalcSize, prgm.regInputSize, prgm.regConstSize, prgm.regConst, prgm.interpreter)

def semantic_introns(evaluate, programm, max_block_size=None, lower_bound=None, test=lambda x,y: x<=y):
    """
        Algorithm 3.2 (elimination of semantic introns) from the book
        arguments:
            evaluate:  a function that take a programm and his fitness
            programm: the programm to evaluate
            max_block_size: max block size to delete , affect the for loop (optionnal)
            lower_bound: Minimum threshold value used to keep the programm, it not provided fitness of the base programm will be used (optionnal)
            test: test that define at which condition a programm will be kept (optionnal, default better fitness than base fitness)
        return:
            Programm
    """
    
    if lower_bound is None:
        base_score = evaluate(programm)
    else:
        base_score = lower_bound

    if max_block_size is None:
        max_block_size = len(programm)
    block_size = 1

    while block_size<max_block_size and block_size<=len(programm):
        cursor = 0
        while cursor<len(programm)-block_size:
            tmp = copy.deepcopy(programm)
            del tmp[cursor:cursor+block_size]
            score = evaluate(tmp)
            if test(base_score, score):
                #base_score = score
                programm = tmp
            else:
                cursor+=1
        block_size+=1
    
    return programm

#Selection
from deap import tools
from operator import attrgetter
from copy import deepcopy
#Diversity tournament -> book Chapter 9 : CONTROL OF DIVERSITY AND VARIATION STEP SIZE
def selDiversityTournament(individuals, k, fitness_size, diversity_size, fitness_first=True, fit_attr="fitness", effective=None):
    assert (1 <= diversity_size <= 2), "Parsimony tournament size has to be in the range [1, 2]."

    def _editDistTournament(individuals, k, select):
        chosen = []
        for i in range(k):
            # Select two individuals from the population
            # The first individual has to be the shortest
            prob = diversity_size / 2.
            inds = select(individuals, k=3)
            if effective:
                tmp = inds
                inds = deepcopy([ind.to_effective(effective)[0] for ind in inds])

            edist12 = edit_distance(inds[0], inds[1])/(max(len(inds[0]), len(inds[1]), 1))
            edist13 = edit_distance(inds[0], inds[2])/(max(len(inds[0]), len(inds[2]), 1))
            edist23 = edit_distance(inds[1], inds[2])/(max(len(inds[1]), len(inds[2]), 1))

            edist = [
                edist12 + edist13,
                edist12 + edist23,
                edist13+ edist23
            ]

            scores = np.argsort(edist)[::-1]
            if edist[scores[0]]==edist[scores[1]]:
                prob=0.5

            if effective:
                inds = tmp
            chosen.append(inds[scores[0]] if random.random() < prob else inds[scores[random.randint(1,2)]])
        
        return chosen
    def _fitTournament(individuals, k, select):
        chosen = []
        for i in range(k):
            aspirants = select(individuals, k=fitness_size)
            chosen.append(max(aspirants, key=attrgetter(fit_attr)))
        return chosen

    if fitness_first:
        tfit = partial(_fitTournament, select=tools.selRandom)
        return _editDistTournament(individuals, k, tfit)
    else:
        tsize = partial(_editDistTournament, select=tools.selRandom)
        return _fitTournament(individuals, k, tsize)
    
#CrossOver
def cxLinear(ind1, ind2, l_min, l_max, l_smax, dc_max, ds_max):
    """
        Linear cross over similar to the one defined in the book chapter 5.7.1 Linear Crossover
        inidiviualq should be list like.
        arguments:
            ind1: first individual to mate
            ind2: second individual to mate
            l_min: minimum length of an individual
            l_max: maximum length of an individual
            l_smax: maximum length of the segment
            dc_max: maximum start difference of segment
            ds_max: maximum size difference of segment
        return:
            mated ind1, ind2
    """
    if len(ind1)>len(ind2):
        ind1, ind2 = ind2, ind1
    if len(ind1)<2: return ind1, ind2
    i1 = random.randint(0, len(ind1)-2)
    i2 = random.randint(0, min(dc_max, len(ind1)-2))
    l1 = random.randint(1, min(len(ind1)-1-i1, l_smax))
    l2 = random.randint(max(1, min(l1-ds_max, len(ind2)-1-i2)), min(len(ind2)-1-i2, l_smax, l1+ds_max))

    if l1 > l2:
        l2 = l1
        if i2 + l2 >= len(ind2): i2 = len(ind2)-1 - l2
    if (len(ind2) - (l2-l1) < l_min) or (len(ind1) - (l2-l1) > l_max):
        l1 = l2 = l1 if random.random() < 0.5 else l2
    if i1 + l1 > len(ind1):
        l1 = l2 = len(ind1) - i1 -1

    s1, s2 = ind1[i1:i1+l1], ind2[i2:i2+l2]
    del ind1[i1:i1+l1]
    del ind2[i2:i2+l2]
    ind1[i1:i1], ind2[i2:i2] = s2, s1

    return ind1, ind2

#MacroMutation
def mutInsert(prog, pConst, pBranch, effective=None):
    idx = random.randint(0, len(prog))
    instr = prog.randomInstruction(prog.regCalcSize, prog.regInputSize, prog.regConstSize, pConst, pBranch, prog.interpreter)
    if effective:
        authorized_dst = set(range(prog.regCalcSize))
        _, R_eff, _ = prog.to_effective(effective, stopAt=idx)
        R_eff.intersection_update(authorized_dst)
        if bool(R_eff):
            instr = prog.Instruction(instr.opcode, random.choice(tuple(R_eff)), instr.inpt1, instr.inpt2)
    prog.insert(idx, instr)
    return prog

def mutDelete(prog, effective=None):
    if len(prog)<2: return prog
    if effective:
        idxs=effective
        if not idxs: return prog
        idx = random.choice(idxs)
    else:
        idx = random.randint(0, len(prog)-1)
    del prog[idx]
    return prog

def mutSwap(prog, effective=None):
    if len(prog)<2: return prog
    if effective:
        idxs=effective
        if len(idxs)>2 and random.random()<0.5:
            i1, i2 = random.sample(idxs, 2)
        elif idxs:
            i1 = random.choice(idxs)
            i2 = random.randint(0, len(prog)-1)
        else:
            return prog
    else:
        i1 = random.randint(0, len(prog)-1)
        i2 = random.randint(0, len(prog)-1)
    prog[i1], prog[i2] = prog[i2], prog[i1]
    return prog