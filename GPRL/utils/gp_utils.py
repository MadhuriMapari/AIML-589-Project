# Copyright (c) Mathurin Videau. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random
from deap import gp
from operator import attrgetter
import operator

operator_complexity = {
    'add': 1, 'substract': 1, 'const': 1, 'multiply': 1, 'divide': 2, 'abs': 2, 'or_': 4, 'and_': 4, 'gt': 4,
    'if_then_else': 5, 'tanh': 4, 'tan': 4,
    'cos': 4, 'sin': 4, 'power': 3, 'cube': 3, 'square': 3, 'sqrt': 3, 'inv': 2, 'log': 3, 'exp': 3
}
#all_features_domain = None#[[-np.inf, np.inf],]
def initfeaturesDomain(all_features_domain,target_indices, objectiveDomain=None):

    target_features_domain = [all_features_domain[idx] for idx in target_indices]
    if objectiveDomain:
        return objectiveDomain + target_features_domain
    else:
        return target_features_domain
def initShape(numTargetFeatures, bin_size, objectiveShape=None):

    if objectiveShape:
        return objectiveShape + [bin_size] * (numTargetFeatures)
    else:
        return [bin_size] * (numTargetFeatures)
def updateAllFeaturesDomain(all_features_domain,currfeaturesValues,numOfFeatures):

    newFeatureDomain = []
    for i in range(len(all_features_domain)):
        dmin, dmax = all_features_domain[i]
        # for faster convergence of domain ratio, we increase maximum value by 25% and decrease minimum value by 25%
        if dmin == -99999 or dmin > currfeaturesValues[i]:
            if currfeaturesValues[i] < 0:
                dmin = currfeaturesValues[i] * (1 + 0.25)
            else:
                dmin = currfeaturesValues[i] * (1 - 0.25)

        if dmax == 99999 or currfeaturesValues[i] > dmax:

            if currfeaturesValues[i] < 0:
                dmax = currfeaturesValues[i] * (1 - 0.25)
            else:
                dmax = currfeaturesValues[i] * (1 + 0.25)
        if dmin == dmax:
            dmin -= dmin / 2
            dmax += dmax / 2
        newFeatureDomain.append((dmin, dmax))

    cFeaturesIdx = np.argsort(np.array(currfeaturesValues))
    return newFeatureDomain, list(np.flip(cFeaturesIdx)[:numOfFeatures])

def getmax_all_features_domain(all_features_domain_lst):
    max_all_features_domain = []
    for all_features_domain in all_features_domain_lst:
        if not max_all_features_domain:
            max_all_features_domain = all_features_domain
        else:
            for i in range(len(all_features_domain)):
                dmin, dmax = all_features_domain[i]
                m_dmin, m_dmax = max_all_features_domain[i]
                if m_dmin > dmin:
                    m_dmin = dmin
                if m_dmax < dmax:
                    m_dmax = dmax
                max_all_features_domain[i] = (m_dmin, m_dmax)
    return max_all_features_domain


def complexity(individual):
    return sum(map(lambda x: operator_complexity.get(x.name, 1), individual))

def div(x1, x2):
    if isinstance(x1, float) or x1.ndim == 0:
        out = np.full_like(x2, 10_000)
    else:
        out = np.full_like(x1, 10_000)
    return np.divide(x1, x2, where=np.abs(x2) > 0.0001, out=out)


def if_then_else(cond, true, false):
    return np.where(cond, true, false)


def classification(*args):
    return np.argmax(args)


def intervales(x):
    if x > 0.33:
        return 1
    elif x < -0.33:
        return -1
    return 0


def exp(x):
    return np.exp(x, where=np.abs(x) < 32, out=np.full_like(x, 100_000))


def log(x):
    return np.log(np.abs(x), where=np.abs(x) > 0.00001, out=np.full_like(x, 10_000))


def power(x, n):
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        if n < 0.0:
            return np.where(np.logical_and(np.abs(x) < 1e6, np.abs(x) > 0.001), np.sign(x) * (np.abs(x)) ** (n), 0.0)
        return np.where(np.abs(x) < 1e6, np.sign(x) * (np.abs(x)) ** (n), 0.0)


def ephemeral_mut(individual, mode, mu=0, std=1):  # mutation of constant
    ephemerals_idx = [index for index, node in enumerate(individual) if isinstance(node, gp.Ephemeral)]
    if len(ephemerals_idx) > 0:
        if mode == "one":
            ephemerals_idx = (random.choice(ephemerals_idx),)
        for i in ephemerals_idx:
            # madhuri For now rejecting the boolean values for mutation
            if individual[i].value == True:
                # for now no mutation will happen
                continue
            new = type(individual[i])()
            new.value = individual[i].value + random.gauss(mu, 0.1 * abs(individual[i].value))
            individual[i] = new
    return individual,


def mutate(individual, expr=None, pset=None, mode="one", mu=0, std=1):  # mutate that include constant and operation modifications
    # mut = gp.mutEphemeral(mut, mode)
    if random.random() < 0.2:
        return ephemeral_mut(individual, mode=mode, mu=mu, std=std)
    if random.random() >= 0.2 and random.random() < 0.5:
        return gp.mutNodeReplacement(individual, pset=pset)
    return gp.mutUniform(individual, expr=expr, pset=pset)


def selQDRandom(grid, k, cellSelection=random.choice):
    idxs = [key for key, v in grid._solutions.items() if v]
    return [cellSelection(grid._solutions[random.choice(idxs)]) for _ in range(k)]


def selFitProp(individuals, fit_attr="fitness"):  # cell fitness proportionnal selection
    min_fit = getattr(min(individuals, key=attrgetter(fit_attr)), fit_attr).values[0]
    sum_fits = sum((getattr(ind, fit_attr).values[0] - min_fit) for ind in individuals)
    u = random.random() * sum_fits
    sum_ = 0
    for ind in individuals:
        sum_ += getattr(ind, fit_attr).values[0] - min_fit
        if sum_ > u:
            return ind
    return ind

def getAllIndivOfContainer(grid,batch_bkp_lst,num_of_obj,target_feature_indices):
    individuals = []
    idxs = [key for key, v in grid._solutions.items() if v]
    for idx in idxs:
        individuals.extend(grid._solutions[idx])

    for indv,feature_vals in batch_bkp_lst:
        if indv in individuals:
            idx = individuals.index(indv)
            mod_indv = individuals[idx]
            featurelist = list(mod_indv.features[:num_of_obj])
            featurelist.extend([feature_vals[k] for k in target_feature_indices])
            mod_indv.features = tuple(featurelist)
            individuals[idx] = mod_indv
            batch_bkp_lst.remove((indv,feature_vals))
            batch_bkp_lst.append((mod_indv,feature_vals))

    return individuals

def updateBatchBkDict(grid, batch_bkp_lst):
    individuals = []
    idxs = [key for key, v in grid._solutions.items() if v]
    for idx in idxs:
        individuals.extend(grid._solutions[idx])
    for indv,fv in batch_bkp_lst:
        if indv not in individuals:
            batch_bkp_lst.remove((indv,fv))
def selQDFitProp(grid, k, cellSelection=selFitProp):
    idxs = [key for key, v in grid._solutions.items() if v]
    return [cellSelection(grid._solutions[min(idxs)]) for _ in range(k)]

def selKQDFitProp(grid, k, cellSelection=selFitProp):
    idxs = [key for key, v in grid._solutions.items() if v]
    return [cellSelection(grid._solutions[random.choice(idxs)]) for _ in range(k)]

def simplifyIndividual(individual, pset):
    simp_indv = individual.copy()
    for tree in simp_indv:
        #print("original tree:", tree)
        for idx in range(len(tree)):
            node = tree[idx]
            # print(" ".join(["\t"]*idx), node.name)
            if isinstance(node, gp.Primitive):
                # get parameters
                argsList = []
                new_node = None
                i = idx
                for j in range(node.arity):
                    if j + i + 1 >= len(tree):
                        break
                    param = tree[j + i + 1]
                    if isinstance(param, gp.Ephemeral):
                        # print('param.value',param.value)
                        argsList.append(param.value)
                        # new_node = type(param)
                        i += 1
                    else:
                        # print('param.name',param.name)
                        argsList = []
                        break
                if len(argsList) == node.arity:
                    # print("found all ephemeral nodes as args [",argsList,"]")#,pset.terminals.values())
                    ret_type = float

                    if node.name in ('div', 'exp', 'log', 'if_then_else', 'power'):
                        val = globals()[node.name](*argsList)  # getattr(gp_utils,node.name)(*argsList)
                        label = "const_" + str(val)
                        ret_type = float
                        # print("########## value of ",node.name," is ",val," and label as ",label," with ret_type  as",ret_type )

                    elif node.name in ('gt', 'and_', 'or_'):
                        val = getattr(operator, node.name)(*argsList)
                        label = "const_" + str(val)
                        ret_type = bool
                        # print("########## value of ",node.name," is ",val," and label as ",label," with ret_type  as",ret_type )
                    else:  # node.name in ('add', 'subtract', 'multiply', 'sin', 'abs', 'tanh', 'tan', 'cos', 'square', 'sqrt', 'inv'):
                        val = getattr(np, node.name)(*argsList)
                        label = "const_" + str(val)
                        ret_type = float
                        # print("########## value of ",node.name," is ",val," and label as ",label," with ret_type  as",ret_type )

                    pset.addEphemeralConstant(label, lambda: val, ret_type)
                    # print("*****-pset.mapping.keys()  =",pset.mapping.keys() )
                    emp_class_key = list(pset.mapping.keys())[-1]
                    # print("*****-emp_class_key  =",emp_class_key )
                    class_ = pset.mapping[emp_class_key]
                    # print("*****-class_  =",class_ )
                    new_node = class_()
                    if new_node:
                        tree[idx] = new_node


