# Copyright (c) Mathurin Videau. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import reduce
from operator import add, itemgetter
import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def convert_logbook_to_dataframe(logbook):
    chapter_keys = logbook.chapters.keys()
    sub_chaper_keys = [c[0].keys() for c in logbook.chapters.values()]

    data = [list(map(itemgetter(*skey), chapter)) for skey, chapter 
                in zip(sub_chaper_keys, logbook.chapters.values())]
    
    tmp = []
    for a in zip(*data):
        flat = []
        for e in a:
            flat+=[*e]
        tmp.append(flat)
    data = np.array(tmp)

    columns = reduce(add, [["_".join([x, y]) for y in s] 
                        for x, s in zip(chapter_keys, sub_chaper_keys)])
    df = pd.DataFrame(data, columns=columns)

    keys = logbook[0].keys()
    data = [[d[k] for d in logbook] for k in keys]
    for d, k in zip(data, keys):
        df[k] = d
    return df

def basic_budget_scheduler(gen_threshold):# list of tuple (generation, simulation_budget)
    def scheduler(gen, population, simulation_budget, parallel_update):
        for g, v in reversed(gen_threshold):
            if gen>g:
                return v, parallel_update
        return simulation_budget, parallel_update
    return scheduler

def save_each_generation(path, modulo=10):
    def save(i, pop, hof, logbook):
        data = {"pop":pop, "hof":hof}  
        if i%modulo == 0:
            with open(os.path.join(path,"data-"+str(i)+".pkl"), "wb") as input_file:
                pickle.dump(data, input_file, pickle.HIGHEST_PROTOCOL)
    return save

def plot_multiObj(logbook,loc):

    gen = logbook.select("gen")

    obj1_maxs = np.array(logbook.chapters["fitness"].select("max"))
    obj1_avgs = np.array(logbook.chapters["fitness"].select("avg"))
    obj1_mins = np.array(logbook.chapters["fitness"].select("min"))

    obj2_maxs = np.array(logbook.chapters["complexity"].select("max"))
    obj2_avgs = np.array(logbook.chapters["complexity"].select("avg"))
    obj2_mins = np.array(logbook.chapters["complexity"].select("min"))

    size_maxs = logbook.chapters["size"].select("max")
    size_avgs = logbook.chapters["size"].select("avg")
    size_mins = logbook.chapters["size"].select("min")

    fig = plt.figure(figsize=(15, 3))

    ax1 = plt.subplot2grid((1,3 ), (0, 0))
    ax1.plot(gen, obj1_maxs, "r-", label="Maximum ")
    ax1.plot(gen, obj1_avgs, "g-", label="Average ")
    ax1.plot(gen, obj1_mins, "b-", label="Minimum ")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel(r"Fitness")
    ax1.set_title("Fitness")

    plt.legend( fontsize=8)

    ax2 = plt.subplot2grid((1, 3), (0, 1))
    ax2.plot(gen, obj2_maxs, "r-", label="Maximum ")
    ax2.plot(gen, obj2_avgs, "g-", label="Average ")
    ax2.plot(gen, obj2_mins, "b-", label="Minimum ")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel(r"Complexity")
    ax2.set_title("Complexity")
    plt.legend( fontsize=8)

    ax3 = plt.subplot2grid((1, 3), (0, 2))
    ax3.plot(gen, size_maxs, "r-", label="Maximum ")
    ax3.plot(gen, size_avgs, "g-", label="Average ")
    ax3.plot(gen, size_mins, "b-", label="Minimum ")
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Size")
    ax3.set_title("Size")

    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(loc+os.sep+'ObjectivePlots.png')
    plt.show()
def plot_ParetoFront(population, halloffame, loc,env='env'):
    #ax4 = plt.subplot2grid((1, 4), (0, 3))
    plt.plot([ind.fitness.values[0] for ind in population],
             [ind.fitness.values[1] for ind in population], "bo", label="Population")
    plt.plot([ind.fitness.values[0] for ind in halloffame],
             [ind.fitness.values[1] for ind in halloffame], "ro", label="HallOfFame")
    plt.plot([ind.fitness.values[0] for ind in halloffame],
             [ind.fitness.values[1] for ind in halloffame], "r-", label="Pareto Frontier", zorder=0, alpha=0.5)
    plt.xlabel(r"Fitness")
    plt.ylabel(r"Complexity")
    plt.title("Multi-Objective Paretofront")

    plt.legend( fontsize=8)

    plt.savefig(loc+os.sep+'ParetoFrontPlot.png')
    plt.show()
