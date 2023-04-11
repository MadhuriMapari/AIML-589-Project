# Copyright (c) Mathurin Videau. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import warnings
import random

import numpy as np
import gym
try:
    import pybullet_envs
except ImportError:
    warnings.warn("PyBullet environment not found")

from deap import creator, tools, base
from GPRL.genetic_programming import linearGP as linGP
from GPRL.utils.utils import plot_multiObj, plot_ParetoFront
from GPRL.UCB import UCBFitness
from GPRL.utils import gp_utils

from GPRL.factory import EvolveFactory
from GPRL.utils.utils import convert_logbook_to_dataframe, save_each_generation

def MC_fitness(individual, n_steps, num_episodes, gamma):
    eff, _, _ = individual.to_effective(list(range(OUTPUT)))
    if len(eff)==0:
        return -2000.0, 999
    if ENV.action_space.shape:
        def agent(inputs):
            register = eff.init_register()
            return eff.execute(eff, inputs, register, list(range(OUTPUT)))
    else:
        if OUTPUT==1:
            def agent(inputs):
                register = eff.init_register()
                return int(eff.execute(eff, inputs, register, list(range(OUTPUT)))>0)
        else:
            def agent(inputs):
                register = eff.init_register()
                return np.argmax(eff.execute(eff, inputs, register, list(range(OUTPUT))))
    s = 0
    steps = 0
    for e in range(num_episodes):
        state = ENV.reset()
        for k in range(n_steps):
            state, reward, done, _ = ENV.step(agent(state))
            s+= gamma*reward
            steps += 1
            if done:
                break
    return s/num_episodes, sum(map(lambda x: linGP.opcode_complexity[x.opcode], eff))

def mutate(individual, pIns, pDel, pSwap, pMut, pConst, pBranch):
    if random.random() < pDel:
        linGP.mutDelete(individual, effective=None)
    if random.random() < pIns:
        linGP.mutInsert(individual, pConst, pBranch, effective=list(range(OUTPUT)))
    if random.random() < pSwap:
        _, _, idxs = individual.to_effective(list(range(OUTPUT)))
        linGP.mutSwap(individual, effective=idxs)
    if random.random() < pMut:
        linGP.Program.mutInstr(individual, 0.3, 0.3, 0.4, pBranch, effective=list(range(OUTPUT)))
    return individual,

class Factory(EvolveFactory):
    def init_global_var(self):
        global ENV, toolbox, creator
        global INPUT, OUTPUT

        ENV = gym.make(self.conf["env"])
        INPUT = ENV.observation_space.shape[0]
        if ENV.action_space.shape:
            OUTPUT = ENV.action_space.shape[0]
        else:
            OUTPUT = ENV.action_space.n

        toolbox, creator = self.make_toolbox()
    
    def make_toolbox(self):
        weights = (1.0, -1.0)
        if self.conf.get("mono-objectif", True) or self.conf.get("mono-objectif", None):
            weights = (1.0,)
        creator.create("FitnessMax", UCBFitness, weights=weights, c=self.conf["c"], sigma=1)
        creator.create("Individual", linGP.Program, fitness=creator.FitnessMax)

        if self.conf['function_set']=="small":
            ops = np.array([True]*3 + [False, True] + [False]*5 + [True]*2)
        elif self.conf['function_set']=="extended":
            ops = np.array([True]*3 + [False, True, False] + [True]*3+ [False] + [True]*2)
        elif self.conf['function_set']=="all":
            ops = np.ones(12, dtype=bool)

        toolbox = base.Toolbox()
        toolbox.register("Program", linGP.initProgam, creator.Individual, regCalcSize=self.conf['regCalcSize'], regInputSize=INPUT, regConstSize=self.conf['regConstSize'], pConst=self.conf['pConst'], pBranch=self.conf["pBranch"], 
                                                                        min_=self.conf['init_size_min'], max_=self.conf['init_size_max'], ops=ops)
        toolbox.register("population", tools.initRepeat, list, toolbox.Program)

        toolbox.register("evaluate", MC_fitness, n_steps=self.conf["n_steps"], num_episodes=self.conf["n_episodes"], gamma=self.conf["gamma"])

        if self.conf.get("selection", False):         
            if self.conf["selection"]["args"]:
                toolbox.register("select", eval(self.conf["selection"]["name"]), **self.conf["selection"]["args"])
            else:
                toolbox.register("select", eval(self.conf["selection"]["name"]))
        else:
            toolbox.register("select", tools.selTournament, tournsize=5)

        toolbox.register("mate", linGP.cxLinear, l_min=1, l_max=3, l_smax=8, dc_max=4, ds_max=8)
        toolbox.register("mutate", mutate, pIns=self.conf["pIns"], pDel=self.conf["pDel"], pSwap=self.conf.get("pSwap", 0.0), pMut=self.conf["pMut"], pConst=self.conf["pConst"], pBranch=self.conf["pBranch"])

        return toolbox, creator

    def get_stats(self):
        stats_fit = tools.Statistics(lambda ind: sum(ind.fitness.rewards)/len(ind.fitness.rewards) if ind.fitness.rewards else ind.fitness.values[0])
        stats_complexity = tools.Statistics(lambda ind: sum(map(lambda x: linGP.opcode_complexity[x.opcode], ind.to_effective(list(range(OUTPUT)))[0])))
        stats_eff = tools.Statistics(lambda ind: len(ind.to_effective(list(range(OUTPUT)))[0]))
        stats_size = tools.Statistics(len)
        #stats_bandit = tools.Statistics(lambda ind: len(ind.fitness.rewards))

        mstats = tools.MultiStatistics(fitness=stats_fit, complexity=stats_complexity, size=stats_size, effective=stats_eff)#, bandit=stats_bandit)
        mstats.register("avg", lambda x: np.mean(x))
        mstats.register("std", lambda x: np.std(x))
        mstats.register("min", lambda x: np.min(x))
        mstats.register("max", lambda x: np.max(x))
        
        return mstats
    
    def close(self):
        global ENV
        ENV.close()

if __name__ == '__main__':
    import multiprocessing

    import multiprocessing
    import argparse
    from deap import algorithms
    import GPRL.algorithms as my_algorithms
    import os
    import pickle
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, help="environment ID", type=str)
    parser.add_argument("--n-episodes", help="Number of episodes", default=1, type=int)
    parser.add_argument("--n-steps", help="Number of step per episode", default=2000, type=int)
    parser.add_argument("--gamma", help="discount factor", default=1.0, type=float)

    parser.add_argument("--regCalcSize", help="Size of calculation register of the program", default=8, type=int)
    parser.add_argument("--regConstSize", help="Size of the constante register of the program", default=10, type=int)
    parser.add_argument("--init-size-min", help="Minimal program size at initialization", default=2, type=int)
    parser.add_argument("--init-size-max", help="Maximum program size at initialization", default=8, type=int)
    parser.add_argument("--pConst", help="Probability of choosing the constante as input", default=1.0, type=float)
    parser.add_argument("--pBranch", help="Probability of choosing a branch operation", default=1.0, type=float)

    parser.add_argument("--pIns", help="macro-mutation probability of insertion", default=0.3, type=float)
    parser.add_argument("--pDel", help="macro-mutation probability of deletion", default=0.6, type=float)
    parser.add_argument("--pSwap", help="macro-mutation probability of swaping instruction", default=0.1, type=float)
    parser.add_argument("--pMut", help="micro-mutation probability of mutating an existing instruction", default=0.5, type=float)

    parser.add_argument("--algorithm", help="algorithm (mu+lambda), (mu, lambda) or UCB", default="UCB", type=str)
    parser.add_argument("--cxpb", help="crossover probability", default=0.0, type=float)
    parser.add_argument("--mutpb", help="mutation probability", default=1.0, type=float)
    parser.add_argument("--lambda_", help="number of offspring", default=100, type=int)
    parser.add_argument("--mu", help="number of parents", default=100, type=int)
    parser.add_argument("--n-gen", help="number of generation", default=100, type=int)
    parser.add_argument("--function-set", help="function set", default="small", type=str)
    parser.add_argument("--simulation-budget", help="number of simulation allowed for UCB", default=1, type=int)
    parser.add_argument("--c", help="constante d'exploration", default=1.0, type=float)
    parser.add_argument("--n-thread", help="number of thread to use", default=1, type=int)
    parser.add_argument("--save-every", help="save hof and population every n generation", default=10, type=int)
    parser.add_argument("--path", help="path to save the results", default=os.path.join("experiments", "results", "linGP"), type=str)

    args = parser.parse_args()

    factory = Factory(vars(args))
    factory.init_global_var()

    mstats = factory.get_stats()
    
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(args.n_thread, initializer=factory.init_global_var)
    toolbox.register("map", pool.map)
    
    pop = toolbox.population(n=args.mu)
    hof = tools.ParetoFront()#tools.HallOfFame(10)

    dir = os.path.join(args.path, "log-linGP-"+ args.env + "-" + os.path.basename(args.path) +"-"+str(time.time()))
    if not os.path.exists(dir):
        os.mkdir(dir)

    if args.algorithm == "UCB":
        algo = partial(my_algorithms.eaMuPlusLambdaUCB, simulation_budget=args.simulation_budget, parallel_update=args.n_thread, iteration_callback=save_each_generation(dir, modulo=args.save_every))
    elif args.algorithm == "(mu, lambda)":
        algo = algorithms.eaMuCommaLambda
    elif args.algorithm == "(mu + lambda)":
        algo = algorithms.eaMuPlusLambda

    pop, log = algo(population=pop, toolbox=toolbox, cxpb=args.cxpb, mutpb=args.mutpb, mu=args.mu, lambda_=args.lambda_, ngen=args.n_gen, stats=mstats, halloffame=hof, verbose=True)

    with open(os.path.join(dir, "pop-final.pkl"), 'wb') as output:
        pickle.dump(list(pop), output, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(dir, "hof-final.pkl"), 'wb') as output:
        pickle.dump(list(hof), output, pickle.HIGHEST_PROTOCOL)

    convert_logbook_to_dataframe(log).to_csv(os.path.join(dir, "log_linGP.csv"), index=False)   
    print("Experiment is saved at : ", dir)
    plot_multiObj(log, dir)
    plot_ParetoFront(pop, hof, dir)
    pool.close()
    factory.close()