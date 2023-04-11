# Copyright (c) Mathurin Videau. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from GPRL.containers.grid import FixGrid as Grid

from GPRL.utils.utils import convert_logbook_to_dataframe, save_each_generation
import experiments.linGP as linGP_script
from GPRL.genetic_programming import linearGP as linGP

def MC_fitness(individual, n_steps, num_episodes, gamma, features_kept):
    eff, _, _ = individual.to_effective(list(range(linGP_script.OUTPUT)))
    
    if len(eff)==0:
        vals = (0,) * np.count_nonzero(features_kept)
        return -200_000.0, *vals
    
    def agent(inputs):
        register = eff.init_register()
        return eff.execute(eff, inputs, register, list(range(linGP_script.OUTPUT)))

    total_features = (0,) * (len(features_kept)-2)
    total_avg_features = (0,) * (len(features_kept)-2)
    s = 0
    for e in range(num_episodes):
        state = linGP_script.ENV.reset()
        for t in range(n_steps):
            state, reward, done, _ = linGP_script.ENV.step(agent(state))

            #Handcrafted Features
            if linGP_script.ENV.unwrapped.spec.id  == "HopperBulletEnv-v0":
                features = [abs(state[-8]), abs(state[-5]), abs(state[-3]), state[-1]]
                #               thight,     leg,            knee          , contact with the ground
            elif linGP_script.ENV.unwrapped.spec.id  == "BipedalWalker-v3":
                features = [abs(state[4]), abs(state[6]), abs(state[9]), abs(state[11]), state[8], state[13]]
                #               hip0,     knee0,         hip1,         , knee1        , contact0, contact1 with the ground
            s+= gamma*reward
            total_features = tuple(x + y for x, y in zip(total_features, features))
            if done:
                break
        total_avg_features = tuple(x/(t+1) + y for x, y in zip(total_features, total_avg_features))
        total_features = (0,) * (len(features_kept)-2)
    
    total_avg_features = [x/num_episodes for x in  total_avg_features]

    nb_weights = len(eff.get_used_regConstIdxs()) 
    total_avg_features.insert(0, nb_weights if nb_weights<=10 else 10)
    
    cpl = sum(map(lambda x: linGP.opcode_complexity[x.opcode], eff))
    if cpl >= 100:
        cpl =100
    total_avg_features.insert(0, cpl)

    total_avg_features = np.array(total_avg_features)

    if s/num_episodes < -200_000.0:
        return -200_000.0, *total_avg_features[features_kept]
    return s/num_episodes, *total_avg_features[features_kept]
           
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
    parser.add_argument("--n-steps", help="Number of step per episode", default=200, type=int)
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

    parser.add_argument("--cxpb", help="crossover probability", default=0.0, type=float)
    parser.add_argument("--mutpb", help="mutation probability", default=1.0, type=float)
    parser.add_argument("--lambda_", help="number of offspring", default=100, type=int)
    parser.add_argument("--batch-size", help="same thing as population size", default=50, type=int)
    parser.add_argument("--n-gen", help="number of generation", default=10, type=int)
    parser.add_argument("--function-set", help="function set", default="small", type=str)
    parser.add_argument("--n-thread", help="number of thread to use", default=1, type=int)
    parser.add_argument("--save-every", help="save hof and population every n generation", default=10, type=int)
    parser.add_argument("--path", help="path to save the results", default=os.path.join("experiments", "results", "qdlinGP"), type=str)

    args = parser.parse_args()
        

    if args.env == "BipedalWalker-v3":
        features_kept = np.zeros(8, dtype=bool)
        features_kept[3] = True
        features_kept[5] = True

        nbBins = [10, 10]      # The number of bins of the grid of elites. Here, we consider only $nbFeatures$ features with $maxTotalBins^(1/nbFeatures)$ bins each
        features_domain = np.array([(0, 100), (0., 10.0), (0., 2.5), (0., 2.5), (0., 2.5),  (0., 2.5), (0., 1.0), (0., 1.0)]) # The domain (min/max values) of the features
        fitness_domain = ((-200.0, 350.0),)              # The domain (min/max values) of the fitness
        max_items_per_bin = 10   # The number of items in each bin of the grid
    elif  args.env != "HopperBulletEnv-v3":
        features_kept = np.zeros(6, dtype=bool)
        features_kept[3] = True
        features_kept[4] = True
        features_kept[5] = True

        nbBins = [10, 10, 10]
        features_domain = np.array([(0, 100), (0., 10.), (0.0, 1.2), (0.0, 1.2), (0.0, 1.2), (0.0, 1.0)])
        fitness_domain = ((-200_000.0, 1100.0),)             
        max_items_per_bin = 5
    else:
        raise ValueError("Environment not supported ! Please use env-id : BipedalWalker-v3 or HopperBulletEnv-v0")
    
    args.features_kept = features_kept
    args.c = 0
    conf = vars(args)
    factory = linGP_script.Factory(conf)
    factory.init_global_var()

    mstats = factory.get_stats()

    pool = multiprocessing.Pool(args.n_thread, initializer=factory.init_global_var)
    linGP_script.toolbox.register("map", pool.map)
    linGP_script.toolbox.register('evaluate', MC_fitness, n_steps=conf["n_steps"], num_episodes=conf["n_episodes"], gamma=conf["gamma"], features_kept=conf["features_kept"])
    
    pop = linGP_script.toolbox.population(n=args.batch_size*10)
    
    grid = Grid(shape=nbBins, max_items_per_bin=max_items_per_bin, fitness_domain=fitness_domain, features_domain=features_domain[features_kept], storage_type=list)
    
    dir = os.path.join(args.path, "log-qdlinGP-"+ args.env +"-"+str(time.time()))
    if not os.path.exists(dir):
        os.mkdir(dir)

    pop, log = algo.qdLambda(pop, linGP_script.toolbox, grid, args.batch_size, cxpb=args.cxpb, mutpb=args.mutpb, lambda_=args.lambda_, ngen=args.n_gen, stats=mstats, iteration_callback=save_each_generation(dir, modulo=args.save_every), verbose=True)

    with open(os.path.join(dir, "grid-final.pkl"), 'wb') as output:
         pickle.dump(list(grid), output, pickle.HIGHEST_PROTOCOL)

    convert_logbook_to_dataframe(log).to_csv(os.path.join(dir, "log_qdlinGP.csv"), index=False)   
    print("Experiment is saved at : ", dir)

    pool.close()
    factory.close()
