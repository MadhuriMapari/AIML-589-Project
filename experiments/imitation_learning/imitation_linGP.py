# Copyright (c) Mathurin Videau. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from GPRL.genetic_programming import linearGP as linGP

from deap import creator, tools, base

import random

import numpy as np
from sklearn.metrics import mean_squared_error

import gym
import pybulletgym

from experiments.imitation_learning.time_feature import TimeFeatureWrapper
from experiments.imitation_learning.imitation_utils import RingReplayBuffer
 

def initializer(name, seed=None, wrapper=lambda x:x):
    global ENV
    ENV = wrapper(gym.make(name))
    if seed:
        ENV.seed(seed)
        ENV.action_space.seed(seed)
        ENV.observation_space.seed(seed)

ENV = TimeFeatureWrapper(gym.make("AntPyBulletEnv-v0"))

INPUT = ENV.observation_space.shape[0]
OUTPUT = ENV.action_space.shape[0]


from GPRL.UCB import UCBFitness
creator.create("FitnessMin", UCBFitness, weights=(-1.0, 1.0))
creator.create("Individual", linGP.Program, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("Program", linGP.initProgam, creator.Individual, regCalcSize=8, regInputSize=INPUT, regConstSize=32, pConst=0.3, pBranch=0.3, min_=16, max_=128)
toolbox.register("population", tools.initRepeat, list, toolbox.Program)

def check_env():
    return 'ENV' in globals()

def select_data(a_learner, a_demo, delta, n=2):
    add = (np.abs(a_learner-a_demo)**n).sum(axis=1)>delta
    return add

def MC_collect_label(agent_learner, agent_demo, env, n_steps):
    if check_env():
        env = ENV
    data = []
    
    env.seed(0)
    env.action_space.seed(0)
    env.observation_space.seed(0)

    state = env.reset()
    a_learner = []
    for k in range(n_steps):
        a_learner.append(agent_learner(state))
        a_demo, _ = agent_demo(state)
        _state, reward, done, _ = env.step(a_learner[k])
        data.append((state, a_demo, _state, reward))
        state = _state
        if done:
            break
    return a_learner, data

def MC_collect_single(agent, env, n_steps):
    if check_env()  and env is None:
        env = ENV
    data = []
    state = env.reset()
    for k in range(n_steps):
        a_demo, _ = agent(state)
        _state, reward, done, _ = env.step(a_demo)
        data.append((state, a_demo, _state, reward))
        state = _state
        if done:
            break
    return data

def MC_fitness(individual, env, n_steps, num_episodes, gamma):
    if check_env() and env is None:
        env = ENV
    eff, _, _ = individual.to_effective(list(range(OUTPUT)))
    register = eff.init_register()
    agent = lambda inputs: eff.execute(eff, inputs, register, list(range(OUTPUT)))
    s = 0
    steps = 0
    for _ in range(num_episodes):
        state = env.reset()
        for k in range(n_steps):
            state, reward, done, _ = env.step(agent(state))
            s+= gamma*reward
            steps += 1
            if done:
                break
    return s

def mse_loss(individual, data, target):
    eff, _, _ = individual.to_effective(list(range(OUTPUT)))
    register = eff.init_register()
    func = lambda inputs: eff.execute(eff, inputs, register, list(range(OUTPUT)))
    y = func(data)
    if(~np.isfinite(y)).any():
        return -np.inf
    mse = mean_squared_error(target, y)
    return mse

def fitness(individual, data, target, env, n_steps, num_episodes, gamma):
    if check_env() and env is None:
        env = ENV
        env.seed(0)
        env.action_space.seed(0)
        env.observation_space.seed(0)
    return mse_loss(individual, data, target), MC_fitness(individual, env, n_steps, num_episodes, gamma),

toolbox.register("evaluate_MC", MC_fitness, env=None, n_steps=300, num_episodes=1, gamma=0.99)
toolbox.register("evaluate_mse", mse_loss)
toolbox.register("evaluate", fitness, env=None, n_steps=300, num_episodes=1, gamma=0.99)

#toolbox.register("select", tools.selTournament, tournsize=12)
toolbox.register("select", tools.selNSGA2)
#toolbox.register("select", tools.selDoubleTournament, fitness_size=12, parsimony_size=1.2, fitness_first=True)

def mutate(individual, pIns, pDel, pSwap, pMut, pConst, pBranch):
    if random.random() < pIns:
        linGP.mutInsert(individual, pConst, pBranch, effective=list(range(OUTPUT)))
    if random.random() < pDel:
        linGP.mutDelete(individual, effective=None)
    if random.random() < pSwap:
        _, _, idxs = individual.to_effective(list(range(OUTPUT)))
        linGP.mutSwap(individual, effective=idxs)
    if random.random() < pMut:
        linGP.Program.mutInstr(individual, 0.3, 0.3, 0.4, pBranch, effective=list(range(OUTPUT)))
    return individual,

toolbox.register("mate", linGP.cxLinear, l_min=2, l_max=128, l_smax=8, dc_max=8, ds_max=10)
toolbox.register("mutate", mutate, pIns=0.45, pDel=0.55, pSwap=0.2, pMut=0.5, pConst=0.3, pBranch=0.3)

def F(weights, individual, idxs, evaluate):
    individual.regConst[idxs] = weights
    return evaluate(individual)

if __name__== "__main__":
    import multiprocessing
    from deap import algorithms
    from stable_baselines3 import TD3

    demonstrator = TD3.load('experiments/imitation_learning/NN/td3-AntBulletEnv-v0.zip')
    demonstrator.action = partial(demonstrator.predict, deterministic=True)

    dataset = RingReplayBuffer(INPUT, OUTPUT, 20_000)
    data = []
    for _ in range(10):
        data.extend(MC_collect_single(demonstrator.action, ENV, 1000))
    dataset.core_transition(data)

    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(12, initializer=initializer, initargs=("AntPyBulletEnv-v0", 0, TimeFeatureWrapper))
    toolbox.register("map", pool.map)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_eff = tools.Statistics(lambda ind: len(ind.to_effective(list(range(OUTPUT)))[0]))
    stats_size = tools.Statistics(len)
    stats_bandit = tools.Statistics(lambda ind: len(ind.fitness.rewards))

    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, effective=stats_eff, bandit=stats_bandit)
    mstats.register("avg", lambda x: np.mean(x, axis=0))
    mstats.register("std", lambda x: np.std(x, axis=0))
    mstats.register("min", lambda x: np.min(x, axis=0))
    mstats.register("max", lambda x: np.max(x, axis=0))

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (mstats.fields if mstats else [])

    #hof = tools.HallOfFame(10)
    hof = tools.ParetoFront()
    pop = toolbox.population(n=1000)
    
    simulation_budget = 10
    paralelle_update = 12

    data = dataset.get_data()
    fitnesses = toolbox.map(partial(toolbox.evaluate, data=data['s'], target=data['a']), pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.add_reward(fit[1])
        ind.fitness.values = fit[0], ind.fitness.calc_fitness(simulation_budget)#, fit[2]  

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.add_reward(fit[1])
        ind.fitness.values = fit[0], ind.fitness.calc_fitness(simulation_budget)#, fit[2]

    tmp = 0
    max_n = 0
    max_ind = None
    while(tmp < simulation_budget):
        inds = toolbox.select(pop, paralelle_update)
        fitnesses = toolbox.map(toolbox.evaluate_MC, inds)
        for ind, fit in zip(inds, fitnesses):
            ind.fitness.add_reward(fit)
            ind.fitness.values = ind.fitness.values[0], ind.fitness.calc_fitness(simulation_budget)#, ind.fitness.values[2]  
            if (max_n < len(ind.fitness.rewards)):
                max_n, max_ind = len(ind.fitness.rewards), ind
        tmp+=1
    
    pop = toolbox.select(pop, 20)
    pop.append(max_ind)
    hof.update(pop)

    record = mstats.compile(pop) if mstats is not None else {}
    logbook.record(gen=0, nevals=len(pop), **record)
    with open('experiments/imitation_learning/log/log.txt', 'w') as f:
        txt = logbook.stream
        f.write(txt)
        f.write('\n')
    print(txt)
    
    for gen in range(0, 2001):
        offspring = algorithms.varOr(pop, toolbox, 200, cxpb=0.1, mutpb=0.9)

        data = dataset.get_data()
        fitnesses = toolbox.map(partial(toolbox.evaluate, data=data['s'], target=data['a']), offspring)
        
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.reset()
            ind.fitness.add_reward(fit[1])
            ind.fitness.values = fit[0], ind.fitness.calc_fitness(simulation_budget)#, fit[2]  
        
        max_n = 0
        max_ind = None
        for ind in pop:
            ind.fitness.update_offset()
            if (max_n < len(ind.fitness.rewards)):
                max_n, max_ind = len(ind.fitness.rewards), ind
        
        popoff = pop+offspring
        tmp = 0
        while(tmp < simulation_budget):
            inds = toolbox.select(popoff, paralelle_update)
            fitnesses = toolbox.map(toolbox.evaluate_MC, inds)
            for ind, fit in zip(inds, fitnesses):
                ind.fitness.add_reward(fit)
                ind.fitness.values = ind.fitness.values[0], ind.fitness.calc_fitness(simulation_budget)#, ind.fitness.values[2] 
                if (max_n < len(ind.fitness.rewards)):
                    max_n, max_ind = len(ind.fitness.rewards), ind
            tmp+=1

        pop = toolbox.select(popoff, 25)

        hof.update(pop)

        if gen%20 == 0 and gen!=0:
            data = []
            for ind in pop:
                eff, _, _ = ind.to_effective(list(range(OUTPUT)))
                register = eff.init_register()
                agent = lambda inputs: eff.execute(eff, inputs, register, list(range(OUTPUT)))
                a_learner, inter_data = MC_collect_label(agent, demonstrator.action, ENV, 300)
                a_learner, inter_data = np.array(a_learner), np.array(inter_data, dtype=dataset.data_type)
                #inter_data = inter_data[select_data(a_learner, inter_data['a'], 1e-2)]#filter data and keep only diverging actions
                dataset.add_transition(inter_data)

            for ind in pop:
                del ind.fitness.values
            
            data = dataset.get_data()
            fitnesses = toolbox.map(partial(toolbox.evaluate, data=data['s'], target=data['a']), pop)
        
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.add_reward(fit[1])
                ind.fitness.values = fit[0], ind.fitness.calc_fitness(simulation_budget)#, fit[2]  

        record = mstats.compile(pop) if mstats is not None else {}
        logbook.record(gen=gen, nevals=len(pop), **record)
        with open('experiments/imitation_learning/log/log.txt', 'a') as f:
            string = logbook.stream
            f.write(string)
            f.write('\n')
        print(string)

        if gen%10==0:
            import pickle
            with open('experiments/imitation_learning/log/'+'hof-'+str(gen)+'.pkl', 'wb') as output:
                pickle.dump(hof, output, pickle.HIGHEST_PROTOCOL)
            with open('experiments/imitation_learning/log/'+'pop-'+str(gen)+'.pkl', 'wb') as output:
                pickle.dump(pop, output, pickle.HIGHEST_PROTOCOL)
    
    pool.close()
    ENV.close()