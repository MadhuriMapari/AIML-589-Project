# Copyright (c) Mathurin Videau. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import operator
import numpy as np

import gym

from deap import gp

from GPRL.utils import gp_utils
from GPRL.MCTS.MCTS import TreeNMCS

def init(env_name, func_set):
    global ENV, pset
    ENV = gym.make(env_name)

    INPUT = ENV.observation_space.shape[0]

    core_function = [ (np.add, [float]*2, float), (np.subtract, [float]*2, float), (np.multiply, [float]*2, float), (gp_utils.div, [float]*2, float)]
    exp_function = [ (gp_utils.exp, [float], float), (gp_utils.log, [float], float)]
    trig_function = [(np.sin, [float], float)]
    if_function = [ (gp_utils.if_then_else, [bool, float, float], float), (operator.gt, [float, float], bool), (operator.and_, [bool, bool], bool), (operator.or_, [bool, bool], bool) ]

    if func_set  == "small":
        function_set = core_function + if_function
    elif func_set == "extended":
        function_set = core_function + exp_function + trig_function   
        
    if bool(ENV.action_space.shape):
        ret = float
        OUTPUT = ENV.action_space.shape[0]
    else:
        OUTPUT = 1
        if ENV.action_space.n == 2:
            ret = bool
        else:
            ret = int
            classification_func = [(gp_utils.classification, [float]*ENV.action_space.n, int)] #(gp_utils.intervales, [float], int)
            function_set += classification_func

    pset = gp.PrimitiveSetTyped("MAIN", [float]*INPUT, ret)
    for primitive in function_set:
        pset.addPrimitive(*primitive)
    pset.addTerminal(0.1, float)
    for i in range(10):
        pset.addTerminal(float(i), float)
    pset.addTerminal(True, bool)
    pset.addTerminal(1, int)


def fitness(individual, n_steps, gamma):
    if ENV.action_space.shape:
        func = gp.compile(gp.PrimitiveTree(individual), pset=pset)
        agent = lambda *s: [func(*s)]
    else:
        func = gp.compile(gp.PrimitiveTree(individual), pset=pset)
        agent = lambda *s: int(func(*s))
    s = 0
    state = ENV.reset()
    for _ in range(n_steps):
        state, reward, done, _ = ENV.step(agent(*state))
        s+= gamma*reward
        if done:
            return s
    return s

def n_fitness(individual, func, n, map=map):
    arg = [individual for _ in range(n)]
    return sum(map(func, arg))/n

if __name__ == "__main__":
    import argparse
    from multiprocessing import Pool

    parser = argparse.ArgumentParser(description='Nested Monte-Carlo script')
    parser.add_argument("--env", required=True, help="environment ID", type=str)
    parser.add_argument("--n-episodes", help="Number of episodes per playout", default=1, type=int)
    parser.add_argument("--n-steps", help="Number of step per episode per episodes", default=500, type=int)
    parser.add_argument("--gamma", help="discount factor", default=1.0, type=float)
    parser.add_argument("--max-size", help="max size of the tree for playout", default=6, type=int)
    parser.add_argument("--level", help="nested monte carlo level", default=3, type=int)
    parser.add_argument("--n-thread", help="number of thread to use for episode parallelization", default=1, type=int)
    parser.add_argument("--function-set", help="function set", default="small", type=str)
    parser.add_argument("--path", help="path to save the results", default="", type=str)

    args = parser.parse_args()
    
    init(args.env, args.function_set)
    pool = Pool(args.n_thread, initializer=init, initargs=(args.env, args.function_set))
    
    func = partial(fitness, n_steps=args.n_steps, gamma=args.gamma)
    
    evaluate = partial(n_fitness, func=func, n=args.n_episodes, map=pool.map)
    
    nmcs = TreeNMCS(pset, args.max_size, evaluate)
    result = gp.PrimitiveTree(nmcs.run([], [pset.ret], 3))

    print(result)
    pool.close()