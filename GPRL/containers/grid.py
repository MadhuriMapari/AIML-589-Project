# Copyright (c) Mathurin Videau. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import reduce
import operator
import numpy as np
from qdpy.containers import Grid

class FixGrid(Grid):# fix a bug in qdpy 0.1.2.1  with Deep grid (nb_items_per_bin)
    def _init_grid(self) -> None:
        """Initialise the grid to correspond to the shape `self.shape`."""
        self._solutions = {x: [] for x in self._index_grid_iterator()}
        self._nb_items_per_bin = np.zeros(self._shape, dtype=int) #{x: 0 for x in self._index_grid_iterator()}
        self._fitness = {x: [] for x in self._index_grid_iterator()}
        self._features = {x: [] for x in self._index_grid_iterator()}
        self._quality = {x: None for x in self._index_grid_iterator()}
        self._quality_array = np.full(self._shape + (len(self.fitness_domain),), np.nan)
        self._bins_size = [(self.features_domain[i][1] - self.features_domain[i][0]) / float(self.shape[i]) for i in range(len(self.shape))]
        self._filled_bins = 0
        self._nb_bins = reduce(operator.mul, self._shape)
        self.recentness_per_bin = {x: [] for x in self._index_grid_iterator()}
        self.history_recentness_per_bin = {x: [] for x in self._index_grid_iterator()}
        self.activity_per_bin = np.zeros(self._shape, dtype=float)

    def get_best_inds(self,batch_bkp_lst,num_of_obj,target_feature_indices):
        #return self._get_best_inds()
        best_inds = []
        for idx, inds in self.solutions.items():
            if len(inds) == 0:
                continue
            best = inds[0]
            for ind in inds[1:]:
                if ind.fitness.dominates(best.fitness):
                    best = ind
                    best_inds.append(best)
            if best == inds[0]:
                best_inds.append(best)
        for indv, all_feature_vals in batch_bkp_lst:
            if indv in best_inds:
                idx = best_inds.index(indv)
                mod_indv = best_inds[idx]
                featurelist = list(mod_indv.features[:num_of_obj])
                featurelist.extend([all_feature_vals[k] for k in target_feature_indices])
                mod_indv.features = tuple(featurelist)
                best_inds[idx] = mod_indv
                batch_bkp_lst.remove((indv, all_feature_vals))
                batch_bkp_lst.append((mod_indv, all_feature_vals))

        return best_inds

class FlexiGrid(Grid):# fix a bug in qdpy 0.1.2.1  with Deep grid (nb_items_per_bin)
    def _init_grid(self) -> None:
        """Initialise the grid to correspond to the shape `self.shape`."""
        self._solutions = {x: [] for x in self._index_grid_iterator()}
        self._nb_items_per_bin = np.zeros(self._shape, dtype=int) #{x: 0 for x in self._index_grid_iterator()}
        self._fitness = {x: [] for x in self._index_grid_iterator()}
        self._features = {x: [] for x in self._index_grid_iterator()}
        self._quality = {x: None for x in self._index_grid_iterator()}
        self._quality_array = np.full(self._shape + (len(self.fitness_domain),), np.nan)
        self._bins_size = [(self.features_domain[i][1] - self.features_domain[i][0]) / float(self.shape[i]) for i in range(len(self.shape))]
        self._filled_bins = 0
        self._nb_bins = reduce(operator.mul, self._shape)
        self.recentness_per_bin = {x: [] for x in self._index_grid_iterator()}
        self.history_recentness_per_bin = {x: [] for x in self._index_grid_iterator()}
        self.activity_per_bin = np.zeros(self._shape, dtype=float)

    def get_best_inds(self):
        #return self._get_best_inds()
        best_inds = []
        for idx, inds in self.solutions.items():
            if len(inds) == 0:
                continue
            best = inds[0]
            for ind in inds[1:]:
                if ind.fitness.dominates(best.fitness):
                    best = ind
                    best_inds.append(best)
            if best == inds[0]:
                best_inds.append(best)
        return best_inds