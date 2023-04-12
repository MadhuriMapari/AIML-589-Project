import pickle
import yaml
from experiments import gp  as gp_script
if '__main__' == __name__:
    if not "conf_gp" in globals():#test if every thing is loaded to not load it twice
        with open("results/log-Pendulum-v1-conf_sqdgp_Pendulum--2023-04-05-18_53_40_g100/conf.yml") as f:#load the conf associated to the experiments (for env input, output and pset initialization)
            conf_gp = yaml.load(f, Loader=yaml.SafeLoader)

        gp_script.Factory(conf_gp["params"]).init_global_var()

    with open("results/log-Pendulum-v1-conf_sqdgp_Pendulum--2023-04-05-18_53_40_g100/hof-final.pkl", "rb") as input_file:
        hof = pickle.load(input_file)
    print(len(hof))
    best = hof[0]

    for k, tree in enumerate(best):
        print("OUTPUT:", str(k+1), tree)
    print(best.fitness.values, len(best.fitness.rewards))

    from GPRL.utils import gp_utils
    from GPRL.genetic_programming import team

    gp_script.ENV = gp_script.gym.make(conf_gp["params"]["env"])
    s = 0
    if gp_script.ENV.action_space.shape:

        agent = gp_script.toolbox.compile(best)
    else:
        func = gp_script.toolbox.compile(best)
        agent = lambda *s: int(func(*s)[0])
    print(best.fitness.values, len(best.fitness.rewards))
    print("Agent", agent)
    steps = 0
    gp_script.ENV.reset()
    gp_script.ENV.render()
    state = gp_script.ENV.reset()
    for k in range(2000):
        state, reward, done, _ = gp_script.ENV.step(agent(*state))
        gp_script.ENV.render()
        s += reward
        steps += 1
        if done:
            break

    print("End! cumulative rewards:", s, " Done?", done, " nb_steps:", k)
    gp_script.ENV.close()

    import numpy as np
    from deap import gp
    import pygraphviz as pgv

    offset = 0
    n = []
    e = []
    l = {}
    for tree in best:  # multi-output support
        expr = tree
        nodes, edges, labels = gp.graph(expr)
        n += list(np.array(nodes) + offset)
        e += map(tuple, list(np.array(edges) + offset))
        for key in list(labels.keys()):
            l[key + offset] = labels[key]
        offset += np.max(nodes) + 1
    nodes = n
    edges = e
    labels = l

    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("img/spTree-GP.png", prog="dot")