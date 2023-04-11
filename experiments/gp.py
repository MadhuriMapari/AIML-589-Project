# Copyright (c) Mathurin Videau. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import time
import gym

try:
    import pybullet_envs
except ImportError:
    warnings.warn("PyBullet environment not found")

import numpy as np
import random
import operator

from functools import partial
import pandas as pd

from deap import base, creator, tools, gp

from GPRL.genetic_programming import team
from GPRL.utils import gp_utils
from GPRL.utils.utils import convert_logbook_to_dataframe, save_each_generation
from GPRL.factory import EvolveFactory

from GPRL.UCB import UCBFitness


def MC_fitness(individual, n_steps, num_episodes, gamma):
    global ENV, toolbox
    if ENV.action_space.shape:
        agent = toolbox.compile(individual)
    else:
        func = toolbox.compile(individual)
        agent = lambda *s: int(func(*s)[0])
    s = 0
    steps = 0
    for _ in range(num_episodes):
        state = ENV.reset()
        for k in range(n_steps):
            state, reward, done, _ = ENV.step(agent(*state))
            s += gamma * reward
            steps += 1
            if done:
                break
    return s, team.team_complexity(individual, gp_utils.complexity)


class Factory(EvolveFactory):

    def init_global_var(self):
        global ENV, toolbox, creator, pset, all_features_domain, target_feature_indices

        ENV = gym.make(self.conf["env"])
        toolbox, creator, pset = self.make_toolbox()

        numfeatures = ENV.observation_space.shape[0]

        all_features_domain = [[-99999, 99999], ] * numfeatures
        if (numfeatures > 3):
            target_feature_indices = random.sample(range(0, numfeatures), 3)

        else:
            target_feature_indices = list(range(0, numfeatures))

    def make_toolbox(self):
        core_function = [(np.add, [float] * 2, float), (np.subtract, [float] * 2, float),
                         (np.multiply, [float] * 2, float), (gp_utils.div, [float] * 2, float)]
        exp_function = [(gp_utils.exp, [float], float), (gp_utils.log, [float], float)]
        trig_function = [(np.sin, [float], float)]
        if_function = [(gp_utils.if_then_else, [bool, float, float], float), (operator.gt, [float, float], bool),
                       (operator.and_, [bool, bool], bool), (operator.or_, [bool, bool], bool)]
        function_set = core_function
        if self.conf['function_set'] == "small":
            function_set += if_function
        elif self.conf["function_set"] == "extended":
            function_set += exp_function + trig_function + if_function

        INPUT = ENV.observation_space.shape[0]
        if bool(ENV.action_space.shape):
            ret = float
            OUTPUT = ENV.action_space.shape[0]
        else:
            OUTPUT = 1
            if ENV.action_space.n == 2:
                ret = bool
            else:
                ret = int
                classification_func = [
                    (gp_utils.classification, [float] * ENV.action_space.n, int)]  # (gp_utils.intervales, [float], int)
                function_set += classification_func

        pset = gp.PrimitiveSetTyped("MAIN", [float] * INPUT, ret)
        for primitive in function_set:
            pset.addPrimitive(*primitive)

        for k in range(INPUT // 2):
            pset.addEphemeralConstant("const_" + str(k), lambda: np.random.uniform(-20.0, 20.0), float)
        #madhuri: As boolen vaue and unity boyth are constants we separate it form features
        #         and instead identify it as ephemeral constants
        pset.addEphemeralConstant("unitConst", lambda: 1, int)
        pset.addEphemeralConstant("bTrue", lambda: True, bool)

        weights = (1.0, -1.0)
        if self.conf.get("mono_objectif", None):
            weights = (1.0,)
        creator.create("FitnessMax", UCBFitness, weights=weights, c=self.conf["c"], sigma=1)
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
        toolbox.register("team_grow", team.init_team, size=OUTPUT, unit_init=lambda: gp.PrimitiveTree(toolbox.expr()))
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.team_grow)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile_gp", gp.compile, pset=pset)
        toolbox.register("compile", team.team_compile, unit_compile=toolbox.compile_gp)

        toolbox.register("evaluate", MC_fitness, n_steps=self.conf["n_steps"], num_episodes=self.conf["n_episodes"],
                         gamma=self.conf["gamma"])

        if self.conf.get("selection", False):
            if self.conf["selection"]["args"]:
                toolbox.register("select", eval(self.conf["selection"]["name"]), **self.conf["selection"]["args"])
            else:
                toolbox.register("select", eval(self.conf["selection"]["name"]))
        else:
            toolbox.register("select", tools.selNSGA2)

        def cx(x1, x2):
            tmp1, tmp2 = gp.cxOnePoint(x1, x2)
            return gp.PrimitiveTree(tmp1), gp.PrimitiveTree(tmp2)

        toolbox.register("mate", team.fixed_mate, unit_cx=cx)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)

        toolbox.register("mutate_gp", gp_utils.mutate, expr=toolbox.expr_mut, pset=pset, mode="all", mu=0, std=1)
        toolbox.register("mutate", team.mutate, unit_mut=lambda x: gp.PrimitiveTree(toolbox.mutate_gp(x)[0]))

        toolbox.decorate("mate", gp.staticLimit(key=lambda x: team.height(x, operator.attrgetter("height")),
                                                max_value=self.conf.get("tree_max_depth", 17)))
        toolbox.decorate("mutate", gp.staticLimit(key=lambda x: team.height(x, operator.attrgetter("height")),
                                                  max_value=self.conf.get("tree_max_depth", 17)))

        return toolbox, creator, pset

    def get_stats(self):
        stats_fit = tools.Statistics(
            lambda ind: sum(ind.fitness.rewards) / len(ind.fitness.rewards) if ind.fitness.rewards else
            ind.fitness.values[0])
        stats_complexity = tools.Statistics(lambda ind: team.team_complexity(ind, gp_utils.complexity))
        stats_size = tools.Statistics(len)
        # stats_bandit = tools.Statistics(lambda ind: len(ind.fitness.rewards))

        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size,
                                       complexity=stats_complexity)  # , bandit=stats_bandit)
        mstats.register("avg", lambda x: np.mean(x))
        mstats.register("std", lambda x: np.std(x))
        mstats.register("min", lambda x: np.min(x))
        mstats.register("max", lambda x: np.max(x))

        return mstats

    def close(self):
        global ENV
        ENV.close()


if __name__ == "__main__":
    import multiprocessing
    import argparse
    from deap import algorithms
    from GPRL import algorithms as algo
    import os
    import pickle
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, help="environment ID", type=str)
    parser.add_argument("--n-episodes", help="Number of episodes", default=1, type=int)
    parser.add_argument("--n-steps", help="Number of step per episode", default=200, type=int)
    parser.add_argument("--gamma", help="discount factor", default=1.0, type=float)
    parser.add_argument("--algorithm", help="algorithm (mu+lambda), (mu, lambda) or UCB", default="UCB", type=str)
    parser.add_argument("--cxpb", help="crossover probability", default=0.1, type=float)
    parser.add_argument("--mutpb", help="mutation probability", default=0.9, type=float)
    parser.add_argument("--lambda_", help="number of offspring", default=500, type=int)
    parser.add_argument("--mu", help="number of parents", default=500, type=int)
    parser.add_argument("--n-gen", help="number of generation", default=2000, type=int)
    parser.add_argument("--function-set", help="function set", default="small", type=str)
    parser.add_argument("--simulation-budget", help="number of simulation allowed for UCB", default=100, type=int)
    parser.add_argument("--c", help="constante d'exploration", default=1.414, type=float)
    parser.add_argument("--n-thread", help="number of thread to use", default=4, type=int)
    parser.add_argument("--save-every", help="save hof and population every n generation", default=100, type=int)
    parser.add_argument("--path", help="path to save the results", default=os.path.join("experiments", "results", "gp"),
                        type=str)

    args = parser.parse_args()

    factory = Factory(vars(args))
    factory.init_global_var()

    mstats = factory.get_stats()

    multiprocessing.set_start_method('spawn')

    pool = multiprocessing.Pool(args.n_thread, initializer=factory.init_global_var)
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=args.mu)
    hof = tools.ParetoFront()

    dir = os.path.join(args.path, "log-gp-" + args.env + "-" + str(time.time()))
    if not os.path.exists(dir):
        os.mkdir(dir)

    if args.algorithm == "UCB":
        algo = partial(algo.eaMuPlusLambdaUCB, simulation_budget=args.simulation_budget, parallel_update=args.n_thread,
                       iteration_callback=save_each_generation(dir, modulo=args.save_every))
    elif args.algorithm == "(mu, lambda)":
        algo = algorithms.eaMuCommaLambda
    elif args.algorithm == "(mu + lambda)":
        algo = algorithms.eaMuPlusLambda

    pop, log = algo(population=pop, toolbox=toolbox, cxpb=args.cxpb, mutpb=args.mutpb, mu=args.mu, lambda_=args.lambda_,
                    ngen=args.n_gen, stats=mstats, halloffame=hof, verbose=True)

    with open(os.path.join(dir, "pop-final.pkl"), 'wb') as output:
        pickle.dump(list(pop), output, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(dir, "hof-final.pkl"), 'wb') as output:
        pickle.dump(list(hof), output, pickle.HIGHEST_PROTOCOL)

    convert_logbook_to_dataframe(log).to_csv(os.path.join(dir, "log_gp.csv"), index=False)
    print("Experiment is saved at : ", dir)

    pool.close()
    factory.close()