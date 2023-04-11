# Copyright (c) Mathurin Videau. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

class EvolveFactory(ABC):#Base class to produce toolbox in different script and keep multiprocessing support (evolve.py script)
    def __init__(self, conf):
        self.conf = conf
    
    @abstractmethod
    def init_global_var(self):
        pass
    
    @abstractmethod
    def make_toolbox(self):
        pass

    @abstractmethod
    def get_stats(self):
        pass

    @ abstractmethod
    def close(self):
        pass
