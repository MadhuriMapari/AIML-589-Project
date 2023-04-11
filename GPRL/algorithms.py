# Copyright (c) Mathurin Videau. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque
from deap import tools, algorithms
import numpy as np
import heapq
from timeit import default_timer as timer
from .UCB import UCBFitness, HeapWithKey


# /!\ stochastique objective must be placed first !
def eaMuPlusLambdaUCB(population, toolbox, simulation_budget, parallel_update, mu, lambda_, cxpb, mutpb, ngen,
                      select=False, stats=None, halloffame=None, verbose=__debug__, budget_scheduler=None,
                      iteration_callback=None):
    assert all([isinstance(ind.fitness, UCBFitness) for ind in population])

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.add_reward(fit[0])
        ind.fitness.values = ind.fitness.calc_fitness(simulation_budget), *fit[1:]

    if select:
        get_individual = lambda inds: toolbox.select(inds, parallel_update)
    else:
        popoff = HeapWithKey(population, lambda x: -x.fitness.values[0])
        get_individual = lambda _: [popoff.pop() for _ in range(parallel_update)]
    tmp = 0
    while (tmp < simulation_budget):
        inds = get_individual(population)
        fitnesses = toolbox.map(toolbox.evaluate, inds)
        for ind, fit in zip(inds, fitnesses):
            ind.fitness.add_reward(fit[0])
            ind.fitness.values = ind.fitness.calc_fitness(simulation_budget), *fit[1:]
            if not select:
                popoff.push(ind)
        tmp += 1

    population = toolbox.select(population, mu)

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    if iteration_callback is not None:
        iteration_callback(0, population, halloffame, logbook)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        if budget_scheduler is not None:
            simulation_budget, parallel_update = budget_scheduler(gen, population, simulation_budget, parallel_update)
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.reset()
            ind.fitness.add_reward(fit[0])
            ind.fitness.values = ind.fitness.calc_fitness(simulation_budget), *fit[1:]

        for ind in population:
            ind.fitness.update_offset()

        popoff = population + offspring
        if not select:
            popoff = HeapWithKey(popoff, lambda x: -x.fitness.values[0])
            get_individual = lambda _: [popoff.pop() for _ in range(parallel_update)]

        tmp = 0
        while (tmp < simulation_budget):
            inds = get_individual(popoff)
            fitnesses = toolbox.map(toolbox.evaluate, inds)
            for ind, fit in zip(inds, fitnesses):
                ind.fitness.add_reward(fit[0])
                ind.fitness.values = ind.fitness.calc_fitness(simulation_budget), *fit[1:]
                if not select:
                    popoff.push(ind)
            tmp += 1

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
        if iteration_callback is not None:
            iteration_callback(gen, population, halloffame, logbook)

    return population, logbook


# "Deapified" version of :
# Regularized Evolution for Image Classifier Architecture Search ; https://ojs.aaai.org/index.php/AAAI/article/view/4405
# based code from : https://github.com/google-research/google-research/tree/master/evolution/regularized_evolution_algorithm
def regularized_evolution(population, toolbox, mu, lambda_, cxpb, mutpb, cycles, stats=None, halloffame=None,
                          verbose=__debug__):
    """Algorithm for regularized evolution (i.e. aging evolution).

    Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
    Classifier Architecture Search".

    Args:
    cycles: the number of cycles the algorithm should run for.

    Returns:
    history: a list of `Model` instances, representing all the models computed
        during the evolution experiment.
    """
    assert mu <= len(population)
    if isinstance(deque, population):
        remove = lambda pop: pop.popleft()
    else:
        remove = lambda pop: pop.pop(0)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    history = []  # Not used by the algorithm, only used to report results.

    # Carry out evolution in cycles.
    while len(history) < cycles:
        parents = toolbox.select(population, k=mu)

        # Create offspring and store it.
        offspring = algorithms.varOr(parents, lambda_, cxpb, mutpb)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population.extend(offspring)
        history.extend(offspring)

        # Withdrawal of oldest individuals
        for _ in range(len(invalid_ind)):
            remove(population)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=len(history), nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return history, logbook.copy()


# objectif is at fitness.fit[0], fitness.fit[1:] is for features.
def nqdLambda(init_batch, toolbox, container, batch_size, ngen, lambda_, cxpb=0.0, mutpb=1.0, stats=None,
              halloffame=None, verbose=False, show_warnings=False, start_time=None, iteration_callback=None):
    from GPRL.containers.grid import FixGrid as Grid
    from GPRL.utils import gp_utils
    """The simplest QD algorithm using DEAP.
    :param init_batch: Sequence of individuals used as initial batch.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution operators.
    :param batch_size: The number of individuals in a batch.
    :param niter: The number of iterations.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :param show_warnings: Whether or not to show warnings and errors. Useful to check if some individuals were out-of-bounds.
    :param start_time: Starting time of the illumination process, or None to take the current time.
    :param iteration_callback: Optional callback function called when a new batch is generated. The callback function parameters are (iteration, batch, container, logbook).
    :returns: The final batch
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    """

    if start_time is None:
        start_time = timer()
    logbook = tools.Logbook()
    logbook.header = ["iteration", "containerSize", "evals", "nbUpdated"] + (stats.fields if stats else []) + [
        "elapsed"]

    if len(init_batch) == 0:
        raise ValueError("``init_batch`` must not be empty.")

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in init_batch if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit[0],
        ind.features = fit[1:]

    if len(invalid_ind) == 0:
        raise ValueError("No invalid individual found !")

    # Update halloffame
    if halloffame is not None:
        halloffame.update(init_batch)

    # Store batch in container
    nb_updated = container.update(init_batch, issue_warning=show_warnings)
    if nb_updated == 0:
        raise ValueError("No individual could be added to the container !")

    # Compile stats and update logs
    record = stats.compile(container) if stats else {}
    logbook.record(iteration=0, containerSize=container.size_str(), evals=len(invalid_ind), nbUpdated=nb_updated,
                   elapsed=timer() - start_time, **record)
    if verbose:
        print(logbook.stream)
    # Call callback function
    if iteration_callback is not None:
        iteration_callback(0, init_batch, container, logbook)

    # Begin the generational process
    for i in range(1, ngen + 1):
        start_time = timer()
        # Select the next batch individuals
        batch = toolbox.select(container, batch_size)  # can we do  NSGA-II selection

        ## Vary the pool of individuals
        offspring = algorithms.varOr(batch, toolbox, lambda_, cxpb=cxpb, mutpb=mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[0],
            ind.features = fit[1:]

        # Replace the current population by the offspring
        nb_updated = container.update(offspring, issue_warning=show_warnings)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(container)

        # Append the current generation statistics to the logbook
        record = stats.compile(container) if stats else {}
        logbook.record(iteration=i, containerSize=container.size_str(), evals=len(invalid_ind), nbUpdated=nb_updated,
                       elapsed=timer() - start_time, **record)
        if verbose:
            print(logbook.stream)
        # Call callback function
        if iteration_callback is not None:
            iteration_callback(i, batch, container, logbook)

    return container, logbook


def sqdLambda(init_batch, toolbox, container, batch_size, ngen, lambda_, cxpb=0.0, mutpb=1.0, stats=None,
              halloffame=None, verbose=False, show_warnings=False, start_time=None, iteration_callback=None,
              objective_domain=None):  # all_features_domain=None,target_feature_indices=[]):
    from GPRL.containers.grid import FixGrid as Grid
    from GPRL.utils import gp_utils
    from experiments import gp as gp_script
    """The simplest QD algorithm using DEAP.
    :param init_batch: Sequence of individuals used as initial batch.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution operators.
    :param batch_size: The number of individuals in a batch.
    :param niter: The number of iterations.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :param show_warnings: Whether or not to show warnings and errors. Useful to check if some individuals were out-of-bounds.
    :param start_time: Starting time of the illumination process, or None to take the current time.
    :param iteration_callback: Optional callback function called when a new batch is generated. The callback function parameters are (iteration, batch, container, logbook).
    :returns: The final batch
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    """

    if start_time is None:
        start_time = timer()
    logbook = tools.Logbook()
    logbook.header = ["iteration", "containerSize", "evals", "nbUpdated"] + (stats.fields if stats else []) + [
        "elapsed"]

    if len(init_batch) == 0:
        raise ValueError("``init_batch`` must not be empty.")
    # c_shape = container.shape
    # max_items_per_bin = container.max_items_per_bin
    # features_domain = container.features_domain
    # fitness_domain = container.fitness_domain
    batch_bkp_lst = []
    fitnesstoIndices = {}
    fitnesstoFtDomain = {}
    NO_OF_INPUTS = len(gp_script.target_feature_indices)
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in init_batch if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit[0][0],
        features = list(fit[0][1:])
        features.extend([fit[1][k] for k in gp_script.target_feature_indices])
        ind.features = tuple(features)
        # all_features_domain = gp_utils.checkandsaveRange(all_features_domain,fit[1])
        all_features_domain, newtargetFeatureIndices = gp_utils.updateAllFeaturesDomain(
            gp_script.all_features_domain, fit[1], NO_OF_INPUTS)

        batch_bkp_lst.append((ind, fit[1]))
        fitnesstoFtDomain[tuple(fit[1])] = all_features_domain
        fitnesstoIndices[ind.fitness.values] = newtargetFeatureIndices

    target_feature_indices = tuple(gp_script.target_feature_indices)
    all_features_domain = gp_utils.getmax_all_features_domain(list(fitnesstoFtDomain.values()))

    if len(invalid_ind) == 0:
        raise ValueError("No invalid individual found !")

    # Update halloffame
    if halloffame is not None:
        halloffame.update(init_batch)

    # Store batch in container
    nb_updated = container.update(init_batch, issue_warning=show_warnings)
    if nb_updated == 0:
        raise ValueError("No individual could be added to the container !")

    # Compile stats and update logs
    record = stats.compile(container) if stats else {}
    logbook.record(iteration=0, containerSize=container.size_str(), evals=len(invalid_ind), nbUpdated=nb_updated,
                   elapsed=timer() - start_time, **record)
    if verbose:
        print(logbook.stream)
        # print("target_indices:",target_feature_indices)
        # print("all_features_domain:", all_features_domain)

    # Call callback function
    if iteration_callback is not None:
        iteration_callback(0, init_batch, container, logbook)

    # Begin the generational process

    for i in range(1, ngen + 1):
        start_time = timer()
        # Select the next batch individuals
        batch = toolbox.select(container, batch_size)
        ## Vary the pool of individuals
        offspring = algorithms.varOr(batch, toolbox, lambda_, cxpb=cxpb, mutpb=mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[0][0],
            features = list(fit[0][1:])
            features.extend([fit[1][k] for k in target_feature_indices])
            ind.features = tuple(features)
            all_features_domain, newtargetFeatureIndices = gp_utils.updateAllFeaturesDomain(
                all_features_domain, fit[1], NO_OF_INPUTS)

            batch_bkp_lst.append((ind, fit[1]))
            fitnesstoFtDomain[tuple(fit[1])] = all_features_domain
            fitnesstoIndices[ind.fitness.values] = newtargetFeatureIndices
        # Replace the current population by the offspring
        no_of_obj = len(objective_domain)
        index = np.argmax(np.array(list(fitnesstoIndices.keys())))
        next_target_feature_indices = tuple(fitnesstoIndices.values())[index]
        all_features_domain = gp_utils.getmax_all_features_domain(list(fitnesstoFtDomain.values()))

        feature_domain = gp_utils.initfeaturesDomain(all_features_domain, target_feature_indices, objective_domain)

        # best_prev_indivs = container.get_best_inds(batch_bkp_lst, no_of_obj, target_feature_indices)
        prev_indvs = gp_utils.getAllIndivOfContainer(container, batch_bkp_lst, no_of_obj, target_feature_indices)
        # print(best_prev_indivs)
        container = Grid(shape=container.shape, max_items_per_bin=container.max_items_per_bin,
                         features_domain=feature_domain, fitness_domain=container.fitness_domain)

        nb_updated = container.update(prev_indvs, issue_warning=show_warnings)
        # try:
        # nb_updated = container.update(best_prev_indivs, issue_warning=show_warnings)
        # except:
        #    print("best indiv not added to container")
        nb_updated += container.update(offspring, issue_warning=show_warnings)
        gp_utils.updateBatchBkDict(container, batch_bkp_lst)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(container)

        # Append the current generation statistics to the logbook

        record = stats.compile(container) if stats else {}
        logbook.record(iteration=i, containerSize=container.size_str(), evals=len(invalid_ind), nbUpdated=nb_updated,
                       elapsed=timer() - start_time, **record)
        if verbose:
            print(logbook.stream)
            # print("target_indices:",target_feature_indices)
            # print("all_features_domain:", all_features_domain)
        target_feature_indices = next_target_feature_indices  # .copy()
        # Call callback function
        if iteration_callback is not None:
            iteration_callback(i, batch, container, logbook)
    batch_bkp_dict = {}
    return container, logbook


def kqdLambda(init_batch, toolbox, container, batch_size, ngen, lambda_, cxpb=0.0, mutpb=1.0, stats=None,
              halloffame=None, verbose=False, show_warnings=False, start_time=None, iteration_callback=None,
              objective_domain=None, all_features_domain=None, target_feature_indices=[]):
    from GPRL.containers.grid import FlexiGrid as Grid
    from GPRL.utils import gp_utils
    from experiments import gp as gp_script
    """The simplest QD algorithm using DEAP.
    :param init_batch: Sequence of individuals used as initial batch.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution operators.
    :param batch_size: The number of individuals in a batch.
    :param niter: The number of iterations.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :param show_warnings: Whether or not to show warnings and errors. Useful to check if some individuals were out-of-bounds.
    :param start_time: Starting time of the illumination process, or None to take the current time.
    :param iteration_callback: Optional callback function called when a new batch is generated. The callback function parameters are (iteration, batch, container, logbook).
    :returns: The final batch
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    """

    if start_time is None:
        start_time = timer()
    logbook = tools.Logbook()
    logbook.header = ["iteration", "containerSize", "evals", "nbUpdated"] + (stats.fields if stats else []) + [
        "elapsed"]

    if len(init_batch) == 0:
        raise ValueError("``init_batch`` must not be empty.")
    # c_shape = container.shape
    # max_items_per_bin = container.max_items_per_bin
    # features_domain = container.features_domain
    # fitness_domain = container.fitness_domain
    fitnesstoIndices = {}
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in init_batch if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit[0][0],
        ind.features = fit[0][1:]
        all_features_domain = fit[1]
        fitnesstoIndices[ind.fitness.values] = fit[2]

    if len(invalid_ind) == 0:
        raise ValueError("No invalid individual found !")

    # Update halloffame
    if halloffame is not None:
        halloffame.update(init_batch)

    # Store batch in container
    nb_updated = container.update(init_batch, issue_warning=show_warnings)
    if nb_updated == 0:
        raise ValueError("No individual could be added to the container !")

    # Compile stats and update logs
    record = stats.compile(container) if stats else {}
    logbook.record(iteration=0, containerSize=container.size_str(), evals=len(invalid_ind), nbUpdated=nb_updated,
                   elapsed=timer() - start_time, **record)
    if verbose:
        print(logbook.stream)
    # Call callback function
    if iteration_callback is not None:
        iteration_callback(0, init_batch, container, logbook)

    # Begin the generational process
    for i in range(1, ngen + 1):
        start_time = timer()
        # Select the next batch individuals
        batch = toolbox.select(container, batch_size)
        ## Vary the pool of individuals
        offspring = algorithms.varOr(batch, toolbox, lambda_, cxpb=cxpb, mutpb=mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[0][0],
            ind.features = fit[0][1:]
            all_features_domain = fit[1]
            fitnesstoIndices[ind.fitness.values] = fit[2]
        # Replace the current population by the offspring
        no_of_obj = len(objective_domain)
        index = np.argmax(np.array(list(fitnesstoIndices.keys())))
        target_feature_indices = list(fitnesstoIndices.values())[index]

        gp_script.targetFeatureIndices = target_feature_indices
        feature_domain = gp_utils.initfeaturesDomain(all_features_domain, target_feature_indices, objective_domain)
        # best_prev_indivs = container.get_best_inds()
        prev_indvs = gp_utils.getAllIndivOfContainer(container)
        # print(best_prev_indivs)
        container = Grid(shape=container.shape, max_items_per_bin=container.max_items_per_bin,
                         features_domain=feature_domain, fitness_domain=container.fitness_domain)
        nb_updated = container.update(prev_indvs, issue_warning=show_warnings)
        # try:
        #    nb_updated = container.update(best_prev_indivs, issue_warning=show_warnings)
        # except:
        #    print("best indiv not added to container")
        nb_updated += container.update(offspring, issue_warning=show_warnings)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(container)

        # Append the current generation statistics to the logbook
        record = stats.compile(container) if stats else {}
        logbook.record(iteration=i, containerSize=container.size_str(), evals=len(invalid_ind), nbUpdated=nb_updated,
                       elapsed=timer() - start_time, **record)
        if verbose:
            print(logbook.stream)
        # Call callback function
        if iteration_callback is not None:
            iteration_callback(i, batch, container, logbook)
    return container, logbook


def qdLambda(init_batch, toolbox, container, batch_size, ngen, lambda_, cxpb=0.0, mutpb=1.0, stats=None,
             halloffame=None, verbose=False, show_warnings=False, start_time=None, iteration_callback=None):
    """The Baysian optimization QD algorithm using DEAP.
    :param init_batch: Sequence of individuals used as initial batch.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution operators.
    :param batch_size: The number of individuals in a batch.
    :param niter: The number of iterations.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :param show_warnings: Whether or not to show warnings and errors. Useful to check if some individuals were out-of-bounds.
    :param start_time: Starting time of the illumination process, or None to take the current time.
    :param iteration_callback: Optional callback function called when a new batch is generated. The callback function parameters are (iteration, batch, container, logbook).
    :returns: The final batch
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    """
    if start_time is None:
        start_time = timer()
    logbook = tools.Logbook()
    logbook.header = ["iteration", "containerSize", "evals", "nbUpdated"] + (stats.fields if stats else []) + [
        "elapsed"]

    if len(init_batch) == 0:
        raise ValueError("``init_batch`` must not be empty.")

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in init_batch if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit[0],
        ind.features = fit[1:]

    if len(invalid_ind) == 0:
        raise ValueError("No invalid individual found !")

    # Update halloffame
    if halloffame is not None:
        halloffame.update(init_batch)

    # Store batch in container
    nb_updated = container.update(init_batch, issue_warning=show_warnings)
    if nb_updated == 0:
        raise ValueError("No individual could be added to the container !")

    # Compile stats and update logs
    record = stats.compile(container) if stats else {}
    logbook.record(iteration=0, containerSize=container.size_str(), evals=len(invalid_ind), nbUpdated=nb_updated,
                   elapsed=timer() - start_time, **record)
    if verbose:
        print(logbook.stream)
    # Call callback function
    if iteration_callback is not None:
        iteration_callback(0, init_batch, container, logbook)

    # Begin the generational process
    for i in range(1, ngen + 1):
        start_time = timer()
        # Select the next batch individuals
        batch = toolbox.select(container, batch_size)

        ## Vary the pool of individuals
        offspring = algorithms.varOr(batch, toolbox, lambda_, cxpb=cxpb, mutpb=mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[0],
            ind.features = fit[1:]

        # Replace the current population by the offspring
        nb_updated = container.update(offspring, issue_warning=show_warnings)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(container)

        # Append the current generation statistics to the logbook
        record = stats.compile(container) if stats else {}
        logbook.record(iteration=i, containerSize=container.size_str(), evals=len(invalid_ind), nbUpdated=nb_updated,
                       elapsed=timer() - start_time, **record)
        if verbose:
            print(logbook.stream)
        # Call callback function
        if iteration_callback is not None:
            iteration_callback(i, batch, container, logbook)

    return container, logbook