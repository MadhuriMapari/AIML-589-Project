# Copyright (c) Mathurin Videau. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from deap import gp, creator, base, tools
import numpy as np

import random
import operator
import warnings
from functools import partial

import gym
try:
    import pybullet_envs
except ImportError:
    warnings.warn("PyBullet environment not found")

from  GPRL.utils import gp_utils
from GPRL.utils.utils import convert_logbook_to_dataframe, save_each_generation

def MC_fitness(individual, n_steps, num_episodes, gamma):
    agent = toolbox.compile(individual)

    s = 0
    steps = 0
    for _ in range(num_episodes):
        state = env.reset() 
        for k in range(n_steps):
            #state, reward, done, _ = env.step(int(agent(*state)))
            #state, reward, done, _ = env.step([agent(*state)])
            state, reward, done, _ = env.step(agent(*state))
            s+= gamma*reward
            steps += 1
            if done:
                break
    return s,

def compile(expr, psets, compile, outputs):# Here same pset for all output, redifining compilation for multi-ouptut support
    adfdict = {}
    func = None
    for pset, subexpr in reversed(list(zip(psets[1:], expr[outputs:]))):
        pset.context.update(adfdict)
        func = compile(subexpr, pset)
        adfdict.update({pset.name: func})
    pset = psets[0]
    pset.context.update(adfdict)
    funcs = [ compile(outexpr, pset) for outexpr in expr[:outputs] ]
    return lambda *args: [f(*args) for f in funcs]



def main(conf):
    def check_output_depth(ind):# Impose output trees very sort depth to ensure use of ADF 
        for tree in ind[:OUTPUT]:
            if tree.height > 3:
                return True
        return False
    
    env = gym.make(conf['env'])

    INPUT = env.observation_space.shape[0]
    OUTPUT = env.action_space.shape[0]
    NUM_ADF = conf['num_ADF']

    feature_ADF = []
    for k in range(NUM_ADF):#ADF function set definition (only 1 level)
        adf =  gp.PrimitiveSetTyped("ADF"+str(k), [float]*conf["num_args"], float)

        adf.addPrimitive(np.add, [float, float], float)
        adf.addPrimitive(np.subtract, [float, float], float)
        adf.addPrimitive(np.multiply, [float, float], float)
        adf.addPrimitive(gp_utils.div, [float, float], float)

        adf.addPrimitive(np.sin, [float], float)
        adf.addPrimitive(gp_utils.exp, [float], float)
        adf.addPrimitive(gp_utils.log, [float], float)
        
        adf.addEphemeralConstant("const_"+str(k), lambda: np.random.uniform(-10.0, 10.0), float)

        feature_ADF.append(adf)
    
    pset = gp.PrimitiveSetTyped("MAIN", [float]*INPUT, float)

    pset.addPrimitive(operator.or_, [bool, bool], bool)
    pset.addPrimitive(operator.and_, [bool, bool], bool)
    pset.addPrimitive(operator.gt, [float, float], bool)
    pset.addPrimitive(gp_utils.if_then_else, [bool, float, float], float)
    pset.addTerminal(True, bool)

    pset.addPrimitive(np.add, [float, float], float)
    pset.addPrimitive(np.subtract, [float, float], float)
    pset.addPrimitive(np.multiply, [float, float], float)

    pset.addEphemeralConstant("const", lambda: np.random.uniform(-10.0, 10.0), float)

    for adf in feature_ADF:
        pset.addADF(adf)

    psets = [pset] + feature_ADF

    from GPRL.UCB import UCBFitness
    creator.create("FitnessMin", UCBFitness, weights=(1.0,), c=2, sigma=5.0)
    creator.create("Tree", gp.PrimitiveTree)

    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    gen = []
    for k, adf in enumerate(feature_ADF):
        toolbox.register('adf_expr'+str(k), gp.genFull, pset=adf, min_=1, max_=5)

    toolbox.register('main_expr', gp.genHalfAndHalf, pset=pset, min_=1, max_=3)

    for k in range(NUM_ADF):
        toolbox.register('ADF'+str(k), tools.initIterate, creator.Tree, getattr(toolbox, 'adf_expr'+str(k)))

    toolbox.register('MAIN', tools.initIterate, creator.Tree, toolbox.main_expr)

    func_cycle = [toolbox.MAIN]*OUTPUT + [ getattr(toolbox, "ADF"+str(k)) for k in range(NUM_ADF) ]# Using adf as feature extraction

    toolbox.register('individual', tools.initCycle, creator.Individual, func_cycle)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register('compile', compile, psets=psets, compile=gp.compile, outputs=OUTPUT)
    toolbox.register("evaluate", MC_fitness, n_steps=conf["n_steps"], num_episodes=conf["n_episodes"], gamma=conf["gamma"])
    toolbox.register('select', tools.selTournament, tournsize=7)

    def mutate(individual, expr, mute, outputs):#Same pset for each outputs
        for i, tree in enumerate(individual[:outputs]):
            if random.random() < conf['mutpb']:
                    individual[i], = mute(individual=tree, expr=expr, pset=psets[0])
        for i, (tree, pset) in enumerate(zip(individual[outputs:], psets[1:])):
                if random.random() < conf['mutpb']:
                    individual[i+outputs], = mute(individual=tree, expr=expr, pset=pset)
        return individual,

    def mate(ind1, ind2, cx):
        for i, (tree1, tree2) in enumerate(zip(ind1, ind2)):
                if random.random() < conf['cxpb']:
                    ind1[i], ind2[i] = cx(tree1, tree2)
        return ind1, ind2

    toolbox.register('mate', mate, cx=gp.cxOnePoint)
    toolbox.register('expr', gp.genFull, min_=1, max_=2)
    toolbox.register('mutate', mutate, expr=toolbox.expr, mute=partial(gp_utils.mutate, mode='all'), outputs=OUTPUT)

    funcs = []
    constrain = partial(gp.staticLimit, max_value=0)
    if True:
        funcs.append(check_output_depth) # Maximum length
    for func in funcs:
        for variation in ["mate", "mutate"]:
            toolbox.decorate(variation, constrain(func))
    return toolbox, creator, env

def initializer(conf):
    global creator, toolbox, env

    toolbox, creator, env = main(conf)

if __name__ == "__main__":
    import multiprocessing
    import argparse
    import os
    import time
    from deap import algorithms
    from GPRL import algorithms as algo
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, help="environment ID", type=str)
    parser.add_argument("--n-episodes", help="Number of episodes", default=1, type=int)
    parser.add_argument("--n-steps", help="Number of step per episode", default=2000, type=int)
    parser.add_argument("--gamma", help="discount factor", default=1.0, type=float)
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
    parser.add_argument("--num-ADF", help="number of ADF to use", default=5, type=int)
    parser.add_argument("--num-args", help="number of argument for each ADF", default=4, type=int)
    parser.add_argument("--save-every", help="save hof and population every n generation", default=10, type=int)
    parser.add_argument("--path", help="path to save the results", default=os.path.join("experiments", "results", "ADF"), type=str)

    args = parser.parse_args()
    conf = vars(args)
    initializer(conf)

    multiprocessing.set_start_method('spawn')

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_size = tools.Statistics(len)
    stats_bandit = tools.Statistics(lambda ind: len(ind.fitness.rewards))

    mstats = tools.MultiStatistics(fitness=stats_fit, bandit=stats_bandit)#size=stats_size)
    mstats.register("avg", lambda x: np.mean(x))
    mstats.register("std", lambda x: np.std(x))
    mstats.register("min", lambda x: np.min(x))
    mstats.register("max", lambda x: np.max(x))

    pool = multiprocessing.Pool(args.n_thread, initializer=initializer, initargs=(conf,))
    toolbox.register("map", pool.map)
    
    pop = toolbox.population(n=args.mu)
    hof = tools.ParetoFront()

    dir = os.path.join(args.path, "log-ADF-"+ args.env +"-"+str(time.time()))
    if not os.path.exists(dir):
        os.mkdir(dir)

    if args.algorithm == "UCB":
        algo = partial(algo.eaMuPlusLambdaUCB, simulation_budget=args.simulation_budget, parallel_update=args.n_thread, iteration_callback=save_each_generation(dir, modulo=args.save_every))
    elif args.algorithm == "(mu, lambda)":
        algo = algorithms.eaMuCommaLambda
    elif args.algorithm == "(mu + lambda)":
        algo = algorithms.eaMuPlusLambda

    pop, log = algo(population=pop, toolbox=toolbox, cxpb=args.cxpb, mutpb=args.mutpb, mu=args.mu, lambda_=args.lambda_, ngen=args.n_gen, stats=mstats, halloffame=hof, verbose=True)

    with open(os.path.join(dir, "pop-final.pkl"), 'wb') as output:
        pickle.dump(list(pop), output, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(dir, "hof-final.pkl"), 'wb') as output:
        pickle.dump(list(hof), output, pickle.HIGHEST_PROTOCOL)

    convert_logbook_to_dataframe(log).to_csv(os.path.join(dir, "log_qdgp.csv"), index=False)   
    print("Experiment is saved at : ", dir)

    pool.close()
    env.close()
