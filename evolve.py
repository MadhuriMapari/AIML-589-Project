from GPRL.utils.utils import basic_budget_scheduler, save_each_generation
import numpy as np
import random

if "__main__" == __name__:
    import yaml
    import argparse
    import multiprocessing
    from deap import gp, algorithms
    from GPRL import algorithms as algo
    from GPRL.utils.utils import convert_logbook_to_dataframe
    from GPRL.UCB import UpdateFitnessHof, UpdateFitnessParetoFront
    import os
    import ntpath
    from shutil import copyfile
    import pickle
    from datetime import datetime
    import time

    parser = argparse.ArgumentParser(description='Main programme to launch experiments from yaml configuration file')
    parser.add_argument("--conf", required=True, help="configuration file path", type=str)
    parser.add_argument("--path", help="path to save the results", default="results", type=str)

    args = parser.parse_args()
    start_time = time.time()
    with open(args.conf) as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)
    numOfObjv=3
    if conf.get("seed", None):
        random.seed(conf["seed"])
        np.random.seed(random.randint(0, 4294967295))

    if conf.get("selection", None):
        if conf["selection"]["name"] in ["selQDRandom", "selFitProp", "selQDFitProp"]:
            conf["selection"]["name"] = "gp_utils." + conf["selection"]["name"]
        elif conf["selection"] == "selDiversityTournament":
            conf["selection"]["name"] = "linGP." + conf["selection"]["name"]
        else:
            conf["selection"]["name"] = "tools." + conf["selection"]["name"]
        conf["params"]["selection"] = conf["selection"]

    if "qd" in conf["algorithm"]["name"]:
        conf["params"]["mono_objectif"] = True

    if conf["individual"] == "Tree" or conf["individual"] == "sTree":
        import experiments.gp as evoTool
    elif conf["individual"] == "Linear":
        import experiments.linGP as evoTool

    factory = evoTool.Factory(conf["params"])

    factory.init_global_var()  # Prepare toolbox and creator

    mstats = factory.get_stats()

    multiprocessing.set_start_method('spawn')

    pool = multiprocessing.Pool(conf["params"]["n_thread"], initializer=factory.init_global_var)
    evoTool.toolbox.register("map", pool.map)

    INPUT = evoTool.ENV.observation_space.shape[0]
    OUTPUT = evoTool.ENV.action_space.shape[0]
    if "qd" in conf["algorithm"]["name"]:

        if "nqd" in conf["algorithm"]["name"]:
            from experiments.nqdgp import MC_fitness
            evoTool.toolbox.register('evaluate', MC_fitness, n_steps=conf["params"]["n_steps"],
                                     num_episodes=conf["params"]["n_episodes"], gamma=conf["params"]["gamma"])

        elif "sqd" in conf["algorithm"]["name"]:
            from GPRL.utils import gp_utils
            from experiments.sqdgp import MC_fitness

            objective_domain = conf["population"]["args"]["features_domain"]
            numOfObjv = len(objective_domain)
            numTargetFeatures = len(evoTool.target_feature_indices)
            conf["population"]["args"]["features_domain"] = gp_utils.initfeaturesDomain(evoTool.all_features_domain,
                                                                                        evoTool.target_feature_indices,
                                                                                        objective_domain)
            conf["population"]["args"]["shape"] = gp_utils.initShape(numTargetFeatures, int(conf["population"]["params"]["binSize"]), conf["population"]["args"]["shape"])
            conf["algorithm"]["args"]["objective_domain"] = objective_domain
            #conf["algorithm"]["args"]["all_features_domain"] = evoTool.all_features_domain
            #conf["algorithm"]["args"]["target_feature_indices"] = evoTool.target_feature_indices

            evoTool.toolbox.register('evaluate', MC_fitness, n_steps=conf["params"]["n_steps"],
                                     num_episodes=conf["params"]["n_episodes"], gamma=conf["params"]["gamma"])
        else:
            if conf["individual"] == "Tree":
                from experiments.qdgp import MC_fitness
            elif conf["individual"] == "Linear":
                from experiments.qdlinGP import MC_fitness
            conf["population"]["args"]["features_domain"] = np.array(conf["population"]["args"]["features_domain"])[conf["params"]["features_kept"]]

            evoTool.toolbox.register('evaluate', MC_fitness, n_steps=conf["params"]["n_steps"],
                                     num_episodes=conf["params"]["n_episodes"], gamma=conf["params"]["gamma"],
                                     features_kept=conf["params"]["features_kept"])

        from GPRL.containers.grid import FixGrid as Grid
        conf["algorithm"]["args"]["container"] = Grid(**conf["population"]["args"])

    pop = evoTool.toolbox.population(n=conf["population"]["init_size"])
    # hof = tools.HallOfFame(10)
    hof = UpdateFitnessHof(10, maxsize_arm=10)
    #hof = UpdateFitnessParetoFront()

    dir = os.path.join(args.path, "log-" + conf["params"]["env"] + "-" + ntpath.basename(args.conf)[
                                                                         :-4] + "--" + datetime.today().strftime(
        '%Y-%m-%d-%H_%M_%S'))
    if not os.path.exists(dir):
        os.mkdir(dir)
    copyfile(args.conf, os.path.join(dir, "conf.yml"))

    if conf["algorithm"]["args"].get("budget_scheduler", None):
        conf["algorithm"]["args"]["budget_scheduler"] = basic_budget_scheduler(
            conf["algorithm"]["args"]["budget_scheduler"])
    if conf["algorithm"]["args"].get("save_every", None):

        conf["algorithm"]["args"]["iteration_callback"] = save_each_generation(dir, modulo=conf["algorithm"]["args"][
            "save_every"])

        conf["algorithm"]["args"]["iteration_callback"] = None
        del conf["algorithm"]["args"]["save_every"]

    algorithm = eval(conf["algorithm"]["name"])  # /!\ not good from a security point of view but flexible...
    pop, log = algorithm(pop, evoTool.toolbox, halloffame=hof, stats=mstats, **conf["algorithm"]["args"])

    df = convert_logbook_to_dataframe(log)

    if "qd" not in conf["algorithm"]["name"] :
        print("Re-evaluating best individual on 1000 episodes for unbiased result...")
        print("But maybe a better one could be found manually in the population.")
        print()
        max_eval = len(max(pop, key=lambda ind: len(ind.fitness.rewards)).fitness.rewards)
        best = min([ind for ind in pop if len(ind.fitness.rewards) >= 0.8 * max_eval],
                   key=lambda ind: ind.fitness.values[1])

        results = np.array(evoTool.toolbox.map(evoTool.toolbox.evaluate, [best for _ in range(1000)]))

        if isinstance(best[0], gp.PrimitiveTree):
            for tree in best:
                print(tree)
        else:
            print(best.to_effective(list(range(evoTool.OUTPUT)))[0])
        print(f"cumulative rewards = {results[:, 0].mean()} +- {results[:, 0].std()}")

        row = df.iloc[-1, :].copy()
        row["fitness_max"] = results[:, 0].mean()
        row.iloc[0] += 1
        df = df.append(row)  # adding unbiased fitness on last row and fitness max column

    with open(os.path.join(dir, "pop-final.pkl"), 'wb') as output:
        pickle.dump(list(pop), output, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(dir, "hof-final.pkl"), 'wb') as output:
        pickle.dump(list(hof), output, pickle.HIGHEST_PROTOCOL)

    df.to_csv(os.path.join(dir, "log.csv"), index=False)

    print("Experiment is saved at : ", dir)
    print("time taken is ", round(time.time() - start_time, 2))
    factory.close()
    pool.close()

