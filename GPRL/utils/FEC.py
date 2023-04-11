# Copyright (c) Mathurin Videau. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from functools import reduce
import numpy as np

#code from : https://www.geeksforgeeks.org/lru-cache-in-python-using-ordereddict/
class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        value = self.cache.get(key, None)
        if value:
            self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last = False)

class FEC(object):#Functionnal Equuivalence Checking
    def __init__(self, data: np.ndarray, cache_size: int):
        self.data = data
        self.cache = LRUCache(cache_size)
    
    def _fingerprint(self, func):
        return hash(func(self.data).data.tobytes())
    
    def add_to_cache(self, func, fitness, fingerprint=None):
        if fingerprint is None:
            fingerprint = self._fingerprint(func)
        self.cache.put(fingerprint, fitness)
    
    def fec(self, func, fingerprint=None):
        if fingerprint is None:
            fingerprint = self._fingerprint(func)
        fitness = self.cache.get(fingerprint)
        return fitness
    
    @staticmethod
    def _cartesian_product_transpose(arrays):# Uniform grid as data util
        broadcastable = np.ix_(*arrays)
        broadcasted = np.broadcast_arrays(*broadcastable)
        rows, cols = reduce(np.multiply, broadcasted[0].shape), len(broadcasted)
        dtype = np.find_common_type([a.dtype for a in arrays], [])

        out = np.empty(rows * cols, dtype=dtype)
        start, end = 0, rows
        for a in broadcasted:
            out[start:end] = a.reshape(-1)
            start, end = end, end + rows
        return out.reshape(cols, rows).T
    
    @staticmethod
    def uniform_domain_dataset(num, *domains):# Uniform grid as dataset (if there to many dimensions use pseudo-random point sampled from Low-discrepancy sequence like Sobol (scipy qmc))
        vec = []

        for (min_, max_) in domains:
            vec.append(np.linspace(min_, max_, num))
        return FEC._cartesian_product_transpose(vec)
        
        