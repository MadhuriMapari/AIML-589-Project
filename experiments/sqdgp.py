# Copyright (c) Mathurin Videau. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import operator
import numpy as np
import pandas as pd
import gym
from deap import gp
from GPRL.containers.grid import FixGrid as Grid

from GPRL.utils import gp_utils
from GPRL.utils.utils import convert_logbook_to_dataframe, save_each_generation
from GPRL.genetic_programming import team
import experiments.gp as gp_script
import re

#featureDomain = (-np.inf,np.inf) * (gp_script.INPUT+3)

def MC_fitness(individual, n_steps, num_episodes, gamma):

    agent = gp_script.toolbox.compile(individual)
    NO_OF_INPUTS = gp_script.ENV.observation_space.shape[0]

    avgStateSpaceVals = (0,) * NO_OF_INPUTS
    s = 0
    for e in range(num_episodes):
        state = gp_script.ENV.reset()
        allstateSpaceVals = (0,) * NO_OF_INPUTS
        nb_steps = n_steps
        for t in range(n_steps):
            if np.random.random() > 0.01:
                action = agent(*state)
            else:
                action = gp_script.ENV.action_space.sample()

            state, reward, done, info = gp_script.ENV.step(action)
            s += gamma * reward

            allstateSpaceVals = tuple(x + y for x, y in zip(allstateSpaceVals, state))

            if done:
                break

        avgStateSpaceVals = tuple(x / (t + 1) + y for x, y in zip(allstateSpaceVals, avgStateSpaceVals))


    avgStateSpaceVals = [x / num_episodes for x in avgStateSpaceVals]
    total_avg_features=[]
    nb_weights = 0
    for ind in individual:
        for node in ind:
            if isinstance(node, gp.Ephemeral):
                nb_weights += 1

    if nb_weights > 20:
        nb_weights = 20
    total_avg_features.insert(0, nb_weights)

    cpl = team.team_complexity(individual, gp_utils.complexity)
    if cpl >= 100:
        cpl = 100
    total_avg_features.insert(0, cpl)

    total_avg_features = np.array(total_avg_features)

    if s / num_episodes < -500_000.0:
        return (-500_000.0, *total_avg_features), avgStateSpaceVals
    return (s / num_episodes, *total_avg_features), avgStateSpaceVals


if '__main__' == __name__:
    import multiprocessing
    import argparse
    from deap import algorithms
    from GPRL import algorithms as algo
    import os
    import pickle
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, help="environment ID", type=str)
    parser.add_argument("--n-episodes", help="Number of episodes", default=3, type=int)
    parser.add_argument("--n-steps", help="Number of step per episode", default=50, type=int)
    parser.add_argument("--gamma", help="discount factor", default=1.0, type=float)

    parser.add_argument("--cxpb", help="crossover probability", default=0.0, type=float)
    parser.add_argument("--mutpb", help="mutation probability", default=1.0, type=float)
    parser.add_argument("--batch-size", help="same thing as population size", default=50, type=int)
    parser.add_argument("--lambda_", help="number of offspring", default=50, type=int)
    parser.add_argument("--n-gen", help="number of generation", default=10, type=int)
    parser.add_argument("--function-set", help="function set", default="small", type=str)
    parser.add_argument("--n-thread", help="number of thread to use", default=1, type=int)
    parser.add_argument("--save-every", help="save hof and population every n generation", default=10, type=int)
    parser.add_argument("--path", help="path to save the results",
                        default=os.path.join("experiments", "results", "qdgp"), type=str)
    # parser.add_argument("--mono-objectif", help="Number of fitness values", default=True, type=bool)

    args = parser.parse_args()

    if "BipedalWalker" in args.env:
        features_kept = np.zeros(8, dtype=bool)
        features_kept[3] = True
        features_kept[5] = True

        nbBins = [10,
                  10]  # The number of bins of the grid of elites. Here, we consider only $nbFeatures$ features with $maxTotalBins^(1/nbFeatures)$ bins each
        features_domain = np.array([(0, 100), (0., 2.0), (0., 2.5), (0., 2.5), (0., 2.5), (0., 2.5), (0., 1.0),
                                    (0., 1.0)])  # The domain (min/max values) of the features
        fitness_domain = ((-200.0, 350.0),)  # The domain (min/max values) of the fitness
        max_items_per_bin = 10  # The number of items in each bin of the grid

    elif "HopperBulletEnv" in args.env and args.env != "HopperBulletEnv-v3":
        features_kept = np.zeros(6, dtype=bool)
        features_kept[3] = True
        features_kept[4] = True
        features_kept[5] = True

        nbBins = [10, 10, 10]
        features_domain = np.array([(0, 100), (0., 20.), (0.0, 1.2), (0.0, 1.2), (0.0, 1.2), (0.0, 1.0)])
        fitness_domain = ((-200_000.0, 1100.0),)
        max_items_per_bin = 5
    else:
        raise ValueError("Environment not supported ! Please use env-id : BipedalWalker-v3 or HopperBulletEnv-v0")

    args.c = 0.0
    args.features_kept = features_kept

    conf = vars(args)
    factory = gp_script.Factory(conf)
    factory.init_global_var()

    mstats = factory.get_stats()
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(args.n_thread, initializer=factory.init_global_var)
    gp_script.toolbox.register("map", pool.map)
    gp_script.toolbox.register('evaluate', MC_fitness, n_steps=conf["n_steps"], num_episodes=conf["n_episodes"],
                               gamma=conf["gamma"], features_kept=conf["features_kept"])

    pop = gp_script.toolbox.population(n=args.batch_size * 10)
    #hof = UpdateFitnessHof(10, maxsize_arm=10)
    grid = Grid(shape=nbBins, max_items_per_bin=max_items_per_bin, fitness_domain=fitness_domain,
                features_domain=features_domain[features_kept], storage_type=list)

    dir = os.path.join(args.path, "log-qdgp-" + args.env + "-" + str(time.time()))
    if not os.path.exists(dir):
        os.mkdir(dir)

    pop, log = algo.qdLambda(pop, gp_script.toolbox, grid, args.batch_size, cxpb=args.cxpb, mutpb=args.mutpb,
                             lambda_=args.lambda_, ngen=args.n_gen, stats=mstats,
                             iteration_callback=save_each_generation(dir, modulo=args.save_every), verbose=True)

    with open(os.path.join(dir, "grid-final.pkl"), 'wb') as output:
        pickle.dump(list(grid), output, pickle.HIGHEST_PROTOCOL)

    convert_logbook_to_dataframe(log).to_csv(os.path.join(dir, "log_qdgp.csv"), index=False)
    print("Experiment is saved at : ", dir)

    pool.close()
    factory.close()
