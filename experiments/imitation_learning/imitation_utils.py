# Copyright (c) Mathurin Videau. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

class RingReplayBuffer(object):
    data_type = None
    def __init__(self, state_size, action_size, size, offset=0):
        self.data_type = [('s', 'f4', (state_size,)), ('a', 'f4', (action_size,)), ('next_s','f4', (state_size,)),('r','f4')]
        self.data = np.zeros(size, dtype=self.data_type)
        self.offset = offset
        self.size = size - offset
        self.idx = 0

    def core_transition(self, data):
        self.size += self.offset
        self.offset = len(data)
        for k, item in enumerate(data):
            self.data[k%self.size] = np.array(item, dtype=self.data_type)
        self.size -= self.offset
    
    def add_transition(self, data):
        if isinstance(data, list):
            for item in data:
                self.data[self.offset + (self.idx%self.size)] = np.array(item, dtype=self.data_type)
                self.idx += 1
        else:
            idx = (self.idx%self.size)
            if data.shape[0] < self.size - idx:
                self.data[self.offset + idx:self.offset+ idx + data.shape[0]] = data
            else:
                print("yolo")#, data.shape[0], self.size -idx, )
                self.data[self.offset + idx:] = data[:self.size-idx]
                self.data[self.offset: self.offset + data.shape[0] - self.size + idx] = data[self.size-idx:]
            self.idx += data.shape[0]

    def get_data(self):
        if self.offset + self.idx > self.size:
            return self.data
        else:
            return self.data[:self.offset+self.idx]