# Copyright (c) Mathurin Videau. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from functools import partial

def eval(epsi, F, w):
    return F(w+epsi)

def Open_AI_ES(map, F, weights, n, steps, alpha=0.001, sigma=0.1):#open ai ES that use eval as objectives
    for _ in range(steps):
        epsi = np.random.normal(loc=0.0, scale=sigma, size=(n, weights.size)).reshape((n, *weights.shape))
        evaluate = partial(eval, F=F, w=weights)
        R = np.array(map(evaluate, epsi))
        A = (R - np.mean(R)) / (np.std(R)+ 1e-4)
        weights += (alpha/(n*sigma))*np.dot(A.T, epsi).flatten()
    return weights