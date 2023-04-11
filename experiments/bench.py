# Copyright (c) Mathurin Videau. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from GPRL.UCB import UpdateFitnessHof, selDoubleTournament, UCBFitness
from functools import partial
import operator
import random

from deap import base, creator, tools, algorithms

from GPRL.utils import gp_utils
from GPRL.genetic_programming import team

import numpy as np
# from numba import njit# Used for oneMax problem only so deactivated to not add more dependencies

from GPRL.algorithms import eaMuPlusLambdaUCB

tools.selUCBDoubleTounament = selDoubleTournament


def evalOneMax(individual, DIMS, STD, n_eval=1):
    if STD == 0.0:
        n_eval = 1
    return numba_evalOneMax(np.array(individual, dtype=int), DIMS, STD, n_eval=n_eval),


# @njit
def numba_evalOneMax(individual, DIMS, STD, n_eval):
    std = np.random.normal(0.0, STD, n_eval)
    return (np.sum(individual) / DIMS) + std.mean()


def oneMax(conf, STD, DIMS, N_EVAL):
    DIMS = 2 ** DIMS
    if "UCB" in conf["algorithm"]["name"]:
        creator.create("FitnessMax", UCBFitness, weights=(1.0,), sigma=STD)
    else:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, DIMS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evalOneMax, DIMS=DIMS, STD=STD, n_eval=N_EVAL)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", eval("tools." + conf["population"]["selection"]), **conf["population"]["args"])

    return toolbox, creator


def check_env():
    return 'ENV' in globals()


def MC_fitness(individual, env, n_steps, num_episodes, gamma):
    if check_env() and env is None:
        env = ENV
    agent = toolbox.compile(individual)
    s = 0
    steps = 0
    for _ in range(num_episodes):
        state = env.reset()
        for k in range(n_steps):
            state, reward, done, _ = env.step(agent(*state))
            s += gamma * reward
            steps += 1
            if done:
                break
    return s / num_episodes, team.team_complexity(individual, gp_utils.complexity)


def env(conf, UCB_SIGMA, NUM_EPISODE, tmp=0):
    import gym
    from deap import gp

    global ENV

    ENV = gym.make(conf["problem"]["env"])

    INPUT = ENV.observation_space.shape[0]
    OUTPUT = ENV.action_space.shape[0]

    function_set = [(np.add, [float] * 2, float), (np.subtract, [float] * 2, float), (np.multiply, [float] * 2, float),
                    (gp_utils.div, [float] * 2, float), (np.sin, [float], float),
                    (partial(gp_utils.power, n=0.5), [float], float, 'sqrt'),
                    (partial(gp_utils.power, n=2), [float], float, 'square'),
                    (partial(gp_utils.power, n=3), [float], float, 'cube')]

    pset = gp.PrimitiveSetTyped("MAIN", [float] * INPUT, float)
    for primitive in function_set:
        pset.addPrimitive(*primitive)
    pset.addEphemeralConstant("const" + str(tmp), lambda: np.random.uniform(-10.0, 10.0), float)

    # for k in range(INPUT): # For large input space
    #     pset.addEphemeralConstant("const_"+str(k), lambda: np.random.uniform(-10.0, 10.0), float)

    if "UCB" in conf["algorithm"]["name"]:
        creator.create("FitnessMax", UCBFitness, weights=(1.0, -1.0), sigma=UCB_SIGMA)
    else:
        creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0))

    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=3, max_=8)
    toolbox.register("team_grow", team.init_team, size=OUTPUT, unit_init=lambda: gp.PrimitiveTree(toolbox.expr()))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.team_grow)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile_gp", gp.compile, pset=pset)
    toolbox.register("compile", team.team_compile, unit_compile=toolbox.compile_gp)

    toolbox.register("evaluate", MC_fitness, env=None, n_steps=conf["problem"]["n_step"], num_episodes=NUM_EPISODE,
                     gamma=conf["problem"]["gamma"])

    if conf["population"]["args"]:
        toolbox.register("select", eval("tools." + conf["population"]["selection"]), **conf["population"]["args"])
    else:
        toolbox.register("select", eval("tools." + conf["population"]["selection"]))

    def cx(x1, x2):
        tmp1, tmp2 = gp.cxOnePoint(x1, x2)
        return gp.PrimitiveTree(tmp1), gp.PrimitiveTree(tmp2)

    toolbox.register("mate", team.fixed_mate, unit_cx=cx)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate_gp", gp_utils.mutate, expr=toolbox.expr_mut, pset=pset, mode="all", mu=0, std=1)
    toolbox.register("mutate", team.mutate, unit_mut=lambda x: gp.PrimitiveTree(toolbox.mutate_gp(x)[0]))

    toolbox.decorate("mate", gp.staticLimit(key=lambda x: team.height(x, operator.attrgetter("height")), max_value=17))
    toolbox.decorate("mutate",
                     gp.staticLimit(key=lambda x: team.height(x, operator.attrgetter("height")), max_value=17))

    return toolbox, creator


def initializer(func, seed=None):
    global toolbox
    print("******************************in Initializer")
    toolbox, _ = func()
    if seed:
        np.random.seed(seed)
        random.seed(seed)


if __name__ == "__main__":
    import multiprocessing
    import time
    import pandas as pd
    import os
    import ntpath
    import argparse
    import yaml
    from GPRL.algorithms import eaMuPlusLambdaUCB

    parser = argparse.ArgumentParser(description='Main programm to launch experiments from yaml configuration file')
    parser.add_argument("--conf", required=True, help="configuration file path", type=str)
    parser.add_argument("--path", help="directory for results", default="", type=str)

    args = parser.parse_args()
    multiprocessing.set_start_method('spawn')

    if args.path == "":
        args.path = os.path.join("experiments", "results", "bench", ntpath.basename(args.conf)[:-4])

    with open(args.conf) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    algo = eval(conf["algorithm"]["name"])
    pbm = conf["problem"]["name"]
    nb_runs = conf["problem"]["nb_runs"]
    if conf["seed"]:
        np.random.seed(conf["seed"])
        random.seed(conf["seed"])

    if pbm == "oneMax":
        results = {"best_pop": [], "best_hof": [], "best_arm": [], "std": [], "dim": [],
                   "n_eval": [], "run_number": [], "UCB_sigma": [], "ngen": [], "mu": [], "lambda": [],
                   "n_thread": [], "time": []}  # n_eval or simulation_buget

        ngen = conf["algorithm"]["args"]["ngen"]
        simulation_budget = conf["algorithm"]["args"].get("simulation_budget", [None])
        tmp = 0
        for n_eval in conf["problem"]["n_eval"]:
            for std in conf["problem"]["std"]:
                for idx, dim in enumerate(conf["problem"]["dims"]):
                    for sim_budg in simulation_budget:
                        for run_number in range(nb_runs):
                            tmp += 1
                            func = partial(oneMax, conf=conf, STD=std, DIMS=dim, N_EVAL=n_eval)
                            toolbox, creator = func()

                            pop = toolbox.population(n=conf["population"]["init_size"])
                            if "UCB" in conf["algorithm"]["name"]:
                                hof = UpdateFitnessHof(20, maxsize_arm=20)
                                conf["algorithm"]["args"]["simulation_budget"] = sim_budg
                            else:
                                hof = tools.HallOfFame(20)

                            if isinstance(ngen, list):
                                conf["algorithm"]["args"]["ngen"] = ngen[idx]

                            with multiprocessing.Pool(conf["algorithm"]["n_thread"], initializer=initializer,
                                                      initargs=(func, conf["seed"])) as p:
                                toolbox.register("map", p.map)
                                start = time.time()
                                pop, log = algo(pop, toolbox, halloffame=hof, **conf["algorithm"]["args"])

                                toolbox.unregister("evaluate")
                                toolbox.register("evaluate", evalOneMax, DIMS=2 ** dim, STD=0.0, n_eval=1)
                                pops = [hof, pop, hof.arm_hof] if isinstance(hof, UpdateFitnessHof) else [hof, pop]
                                for l in pops:
                                    fitnesses = toolbox.map(toolbox.evaluate, l)
                                    for ind, fit in zip(l, fitnesses):
                                        ind.fitness.values = fit

                            del creator.Individual
                            del creator.FitnessMax

                            print("Run ", str(tmp))

                            results["time"].append(time.time() - start)
                            results["best_pop"].append(tools.selBest(pop, 1)[0].fitness.values[0])
                            results["best_hof"].append(tools.selBest(hof, 1)[0].fitness.values[0])
                            results["std"].append(std)
                            results["dim"].append(dim)
                            results["ngen"].append(conf["algorithm"]["args"]["ngen"])
                            results["run_number"].append(run_number)
                            results["n_thread"].append(conf["algorithm"]["n_thread"])

                            if "UCB" in conf["algorithm"]["name"]:
                                results["n_eval"].append(sim_budg)
                            else:
                                results["n_eval"].append(n_eval)

                            results["mu"].append(conf["algorithm"]["args"].get("mu", None))
                            results["lambda"].append(conf["algorithm"]["args"].get("lambda", None))

                            results["UCB_sigma"].append(conf["algorithm"].get("UCB_sigma", None))
                            if isinstance(hof, UpdateFitnessHof):
                                results["best_arm"].append(hof.arm_hof[0].fitness.values[0])
                            else:
                                results["best_arm"].append(None)
    elif pbm == "env":

        results = {"best_pop": [], "best_hof": [], "best_arm": [], "env": [], "n_step": [],
                   "num_episode": [], "run_number": [], "UCB_sigma": [], "ngen": [], "mu": [], "lambda": [],
                   "n_thread": [], "complexity_pop": [], "complexity_hof": [], "complexity_arm": [],
                   "time": []}  # num_episode or simulation_budget depending on UCB alg or not

        simulation_budget = conf["algorithm"]["args"].get("simulation_budget", [None])
        tmp = 0

        for num_episode in conf["problem"]["num_episode"]:
            for UCB_sigma in conf["algorithm"].get("UCB_sigma", [None]):
                for sim_budg in simulation_budget:
                    for run_number in range(nb_runs):
                        tmp += 1
                        func = partial(env, conf=conf, UCB_SIGMA=UCB_sigma, NUM_EPISODE=num_episode, tmp=tmp)

                        toolbox, creator = func()

                        pop = toolbox.population(n=conf["population"]["init_size"])
                        if "UCB" in conf["algorithm"]["name"]:
                            hof = UpdateFitnessHof(20, maxsize_arm=20)
                            conf["algorithm"]["args"]["simulation_budget"] = sim_budg
                        else:
                            hof = tools.ParetoFront()

                        with multiprocessing.Pool(conf["algorithm"]["n_thread"], initializer=initializer,
                                                  initargs=(func, conf["seed"])) as p:
                            toolbox.register("map", p.map)
                            start = time.time()
                            pop, log = algo(pop, toolbox, halloffame=hof, **conf["algorithm"]["args"])

                            toolbox.unregister("evaluate")
                            toolbox.register("evaluate", MC_fitness, env=None, n_steps=conf["problem"]["n_step"],
                                             num_episodes=50, gamma=1.0)
                            pops = [hof, pop, hof.arm_hof] if isinstance(hof, UpdateFitnessHof) else [hof, pop]
                            for l in pops:
                                fitnesses = toolbox.map(toolbox.evaluate, l)
                                for ind, fit in zip(l, fitnesses):
                                    del ind.fitness.values
                                    ind.fitness.values = fit

                        del creator.Individual
                        del creator.FitnessMax

                        print("Run ", str(tmp))

                        best_pop = tools.selBest(pop, 1)[0]
                        best_hof = tools.selBest(hof, 1)[0]

                        results["time"].append(time.time() - start)
                        results["best_pop"].append(best_pop.fitness.values[0])
                        results["best_hof"].append(best_hof.fitness.values[0])
                        results["complexity_pop"].append(best_pop.fitness.values[1])
                        results["complexity_hof"].append(best_hof.fitness.values[1])
                        results["ngen"].append(conf["algorithm"]["args"]["ngen"])
                        results["env"].append(conf["problem"]["env"])
                        results["run_number"].append(run_number)
                        results["n_thread"].append(conf["algorithm"]["n_thread"])
                        results["n_step"].append(conf["problem"]["n_step"])

                        if "UCB" in conf["algorithm"]["name"]:
                            results["num_episode"].append(sim_budg)
                        else:
                            results["num_episode"].append(num_episode)

                        results["mu"].append(conf["algorithm"]["args"].get("mu", None))
                        results["lambda"].append(conf["algorithm"]["args"].get("lambda", None))

                        results["UCB_sigma"].append(UCB_sigma if conf["algorithm"].get("UCB_sigma", None) else None)
                        if isinstance(hof, UpdateFitnessHof):
                            best_arm = tools.selBest(hof.arm_hof, 1)[0]
                            results["best_arm"].append(best_arm.fitness.values[0])
                            results["complexity_arm"].append(best_arm.fitness.values[1])
                        else:
                            results["best_arm"].append(None)
                            results["complexity_arm"].append(None)

    pd.DataFrame(results).to_csv(args.path, index=False)
