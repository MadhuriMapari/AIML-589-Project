# Copyright (c) Mathurin Videau. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from deap import gp, creator, base, tools
import numpy as np

import operator
from functools import partial

import gym
import pybulletgym

from stable_baselines3 import TD3

from GPRL.utils import gp_utils
from GPRL.genetic_programming import team
from experiments.imitation_learning.imitation_utils import RingReplayBuffer
from experiments.imitation_learning.time_feature import TimeFeatureWrapper

from sklearn.metrics import mean_squared_error

def check_env():
    return 'ENV' in globals()
    
def select_data(a_learner, a_demo, delta, n=2):
    add = (np.abs(a_learner-a_demo)**n).sum(axis=1)>delta
    return add

def MC_collect_label(agent_learner, agent_demo, env, n_steps):
    if check_env():
        env = ENV
    data = []

    state = env.reset()
    a_learner = []
    for k in range(n_steps):
        a_learner.append(agent_learner(*state))
        a_demo, _ = agent_demo(state)
        _state, reward, done, _ = env.step(a_learner[k])
        data.append((state, a_demo, _state, reward))
        state = _state
        if done:
            break
    return a_learner, data

def MC_collect_label_(agent_learner, agent_demo, env, n_steps):
    if check_env():
        env = ENV
    data = []

    flag = False
    state = env.reset()
    a_learner = []
    for k in range(n_steps):
        a_learner.append(agent_learner(*state))
        a_demo, _ = agent_demo(state)
        if flag:
            _state, reward, done, _ = env.step(a_demo)
            data.append((state, a_demo, _state, reward))
        else:
            _state, reward, done, _ = env.step(a_learner[k])
        if sum(np.abs(a_demo-a_learner[k]))>0.1:
            flag = True
        
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
    agent = toolbox.compile(individual)
    s = 0
    steps = 0
    for _ in range(num_episodes):
        state = env.reset()   
        for k in range(n_steps):
            state, reward, done, _ = env.step(agent(*state))
            s+= gamma*reward
            steps += 1
            if done:
                break
    return s

def mse_loss(individual, data, target):
    func = toolbox.compile(individual)
    y = func(*data)
    for k in range(len(y)):
        if isinstance(y[k], float) or y[k].ndim==0:
            y[k] = np.full(target.shape[1], y[k])
    y = np.array(y)
    if(~np.isfinite(y)).any():
        return -np.inf
    mse = mean_squared_error(target, y)
    return mse


def initializer(name, seed=None, wrapper=lambda x:x):
    global ENV
    ENV = wrapper(gym.make(name))
    if seed:
        ENV.seed(seed)
        #env.action_space.seed(seed)
        #env.observation_space.seed(seed)

ENV = TimeFeatureWrapper(gym.make("AntPyBulletEnv-v0"))

INPUT = ENV.observation_space.shape[0]
OUTPUT = ENV.action_space.shape[0]

core_function = [ (np.add, [float]*2, float), (np.subtract, [float]*2, float), (np.multiply, [float]*2, float), (gp_utils.div, [float]*2, float)]
trig_function = [(np.sin, [float], float)]
exp_function = [ (gp_utils.exp, [float], float), (gp_utils.log, [float], float)]
if_function = [ (gp_utils.if_then_else, [bool, float, float], float), (operator.gt, [float, float], bool), (operator.and_, [bool, bool], bool), (operator.or_, [bool, bool], bool) ]

function_set = core_function + exp_function + if_function + trig_function

pset = gp.PrimitiveSetTyped("MAIN", [float]*INPUT, float)
for primitive in function_set:
    pset.addPrimitive(*primitive)

for i in range(INPUT//2):# Force the use of more constante.
    pset.addEphemeralConstant("const_"+str(i), lambda: np.random.uniform(-20, 20), float)
pset.addTerminal(True, bool)


from GPRL.UCB import UCBFitness
creator.create("FitnessMin", UCBFitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr",  gp.genHalfAndHalf, pset=pset, min_=3, max_=8)
toolbox.register("team_grow", team.init_team, size=OUTPUT, unit_init=lambda: gp.PrimitiveTree(toolbox.expr()))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.team_grow)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile_gp", gp.compile, pset=pset)
toolbox.register("compile", team.team_compile, unit_compile=toolbox.compile_gp)

def fitness(individual, data, target, env, n_steps, num_episodes, gamma):
    if check_env() and env is None:
        env = ENV
    return mse_loss(individual, data, target),  MC_fitness(individual, env, n_steps, num_episodes, gamma)#, team.team_complexity(individual, gp_utils.complexity) 

toolbox.register("evaluate_MC", MC_fitness, env=None, n_steps=300, num_episodes=1, gamma=0.99)
toolbox.register("evaluate", fitness, env=None, n_steps=300, num_episodes=1, gamma=0.99)

#toolbox.register("select", tools.selTournament, tournsize=12)
toolbox.register("select", tools.selNSGA2)
#toolbox.register("select", tools.selDoubleTournament, fitness_size=12, parsimony_size=1.3, fitness_first=True)

def cx(x1, x2):
    tmp1, tmp2 = gp.cxOnePoint(x1, x2)
    return gp.PrimitiveTree(tmp1), gp.PrimitiveTree(tmp2)
toolbox.register("mate", team.fixed_mate, unit_cx=cx)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate_gp", gp_utils.mutate, expr=toolbox.expr_mut, pset=pset, mode="all", mu=0, std=1)
toolbox.register("mutate", team.mutate, unit_mut=lambda x: gp.PrimitiveTree(toolbox.mutate_gp(x)[0]))

toolbox.decorate("mate", gp.staticLimit(key=lambda x: team.height(x, operator.attrgetter("height")), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=lambda x: team.height(x, operator.attrgetter("height")), max_value=17))

if __name__== "__main__":
    import multiprocessing
    from deap import algorithms

    print("here:0")
    demonstrator = TD3.load('D:/MAI/AIML589/gpxrl-master/experiments/imitation_learning/NN/td3-AntBulletEnv-v0')
    demonstrator.action = partial(demonstrator.predict, deterministic=True)
    print("here:1")
    dataset = RingReplayBuffer(INPUT, OUTPUT, 20_000)
    data = []
    print("here:1-2")
    for _ in range(10):
        data.extend(MC_collect_single(demonstrator.action, ENV, 300))
    dataset.core_transition(data)
    print("here:2")
    pool = multiprocessing.Pool(14)
    toolbox.register("map", pool.map)
    print("here:3")
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    stats_bandit = tools.Statistics(lambda ind: len(ind.fitness.rewards))
    print("here:4")
    multiprocessing.set_start_method('spawn', force=True)
    #try:
    #    multiprocessing.set_start_method('spawn')
    #except RuntimeError:
    #    pass
    print("here:5")
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, bandit=stats_bandit)
    mstats.register("avg", lambda x: np.mean(x, axis=0))
    mstats.register("std", lambda x: np.std(x, axis=0))
    mstats.register("min", lambda x: np.min(x, axis=0))
    mstats.register("max", lambda x: np.max(x, axis=0))
    print("here:6")
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (mstats.fields if mstats else [])
    print("here:7")
    #hof = tools.HallOfFame(10)
    hof = tools.ParetoFront()
    pop = toolbox.population(n=1000)
    print("here:8")
    simulation_budget = 10
    paralelle_update = 12

    data = dataset.get_data()
    fitnesses = toolbox.map(partial(toolbox.evaluate, data=data['s'].T, target=data['a'].T), pop)
    print("here:9")
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.add_reward(fit[1])
        ind.fitness.values = fit[0], ind.fitness.calc_fitness(simulation_budget)#, fit[2]  
    print("here:10")
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
    print("here:11")
    pop = toolbox.select(pop, 2001)
    hof.update(pop)
    print("here:12")
    record = mstats.compile(pop) if mstats is not None else {}
    logbook.record(gen=0, nevals=len(pop), **record)
    print("here:13")
    with open('D:/MAI/AIML589/gpxrl-master/experiments/imitation_learning/log/log_gp.txt', 'w') as f:
        txt = logbook.stream
        f.write(txt)
        f.write('\n')
    print("from here:1", txt)
    
    for gen in range(0, 2001):
        offspring = algorithms.varOr(pop, toolbox, 200, cxpb=0.0, mutpb=1.0)

        data = dataset.get_data()
        fitnesses = toolbox.map(partial(toolbox.evaluate, data=data['s'].T, target=data['a'].T), offspring)
        
        for ind, fit in zip(offspring, fitnesses):
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
                #ind.fitness.values = ind.fitness.calc_fitness(simulation_budget),
                ind.fitness.values = ind.fitness.values[0], ind.fitness.calc_fitness(simulation_budget)#, ind.fitness.values[2] 
                if (max_n < len(ind.fitness.rewards)):
                    max_n, max_ind = len(ind.fitness.rewards), ind
            tmp+=1

        pop = toolbox.select(pop + offspring, 20)

        hof.update(pop)

        if gen%10 == 0 and gen!=0:
            data = []
            for ind in tools.selBest(pop, 5):
                #a_learner, inter_data = MC_collect_label(toolbox.compile(ind), demonstrator.action, ENV, 300)
                a_learner, inter_data = MC_collect_label_(toolbox.compile(ind), demonstrator.action, ENV, 300)
                a_learner, inter_data = np.array(a_learner), np.array(inter_data, dtype=dataset.data_type)
                #inter_data = inter_data[select_data(a_learner, inter_data['a'], 1e-2)]#Only keep diverging actions
                dataset.add_transition(inter_data)

            for ind in pop:
                del ind.fitness.values
            
            data = dataset.get_data()
            fitnesses = toolbox.map(partial(toolbox.evaluate, data=data['s'].T, target=data['a'].T), pop)        
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.add_reward(fit[1])
                ind.fitness.values = fit[0], ind.fitness.calc_fitness(simulation_budget)#, fit[2]  

        record = mstats.compile(pop) if mstats is not None else {}
        logbook.record(gen=gen, nevals=len(pop), **record)
        with open('experiments/imitation_learning/log/log_gp.txt', 'a') as f:
            string = logbook.stream
            f.write(string)
            f.write('\n')
        print("from here:2", string)

        if gen%10==0:
            import pickle
            with open('experiments/imitation_learning/log/'+'hof-'+str(gen)+'.pkl', 'wb') as output:
                pickle.dump(hof, output, pickle.HIGHEST_PROTOCOL)
            with open('experiments/imitation_learning/log/'+'pop-'+str(gen)+'.pkl', 'wb') as output:
                pickle.dump(pop, output, pickle.HIGHEST_PROTOCOL)
    print("end here")
    pool.close()
    ENV.close()
    

