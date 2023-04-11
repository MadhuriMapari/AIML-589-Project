# Copyright (c) Mathurin Videau. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
from sklearn.linear_model import LinearRegression

"""
This module define all the functions to evolve individual in groups.
Each individual of the group is an output.
"""

class MGGP(list):#Multi-gene genetic programming
    def __init__(self, content):
        list.__init__(self, content)
        self.linear = None
    
    def fit(self, func, x, target):
        features = func(*x).T
        self.linear = LinearRegression()
        self.linear = self.linear.fit(features, y=target)
    
    def predict(self, func, x):
        return self.linear.predict(func(*x).T)
   
def mutate(team, unit_mut):#Randomly mute an individual of the team
    idx = random.randint(0, len(team)-1)
    team[idx] = unit_mut(team[idx])
    return team,

def fixed_mate(team1, team2, unit_cx):#Randomly mate two individuals that has the same position in the team
    assert len(team1)==len(team2)
    idx = random.randint(0, len(team1)-1)
    team1[idx], team2[idx] = unit_cx(team1[idx], team2[idx])
    return team1, team2

def cx_low_level(team1, team2, unit_cx):#crossover between any individuals of the two groups
    idx1 = random.randint(0, len(team1)-1)
    idx2 = random.randint(0, len(team2)-1)
    team1[idx1], team2[idx2] = unit_cx(team1[idx1], team2[idx2])
    return team1, team2

def cx_hight_level(team1, team2, cxprb):#exchange gene from each team
    add_team1 = []
    k = 0
    while k<len(team2):
        if random.random() < cxprb:
            add_team1.append(team2.pop(k))
            k-=1
        k+=1
    k=0
    team2
    while k<len(team1):
        if random.random() < cxprb:
            team2.append(team1.pop(k))
            k-=1
        k+=1
    team1.extend(add_team1)
    if not team1:
        team1.append(team2[-1])
    elif not team2:
        team2.append(team1[-1])
    return team1, team2

def mutation_del(team):#delete an individual in the team
    team.pop(random.randint(0, len(team)-1))
    return team

def init_team(size, unit_init):
    team = [unit_init() for _ in range(size)]
    return team

def init_randomSize_team(max_size, unit_init):
    size = random.randint(1, max_size-1)
    return init_team(size, unit_init)

def team_complexity(team, unit_complexity):
    return sum(map(unit_complexity, team))

def team_size_constraint(team, size):
    while len(team)>= size:
        mutation_del(team)
    return team

def team_compile(team, unit_compile):#compile each individual in the the team
    funcs = list(map(unit_compile, team))
    def func(*args):
        return [f(*args) for f in funcs]
    return func

def height(team, op):
    return max(map(op, team))