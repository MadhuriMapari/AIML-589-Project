# eXplainable GP-based RL Policy
A python implementation of symbolic policy for interpretable reinforcement learning using genetic programming.  
## Setup
### Requirements
- numpy
- deap
- qdpy
- pygraphviz (for easier understanding of program)

### Installing dependencies
- clone this repo
- install with ` python -m pip install -r requirement.txt ` for base installation (no pygraphiz)
- install with ` python -m pip install -r requirement_with_pygrphivz.txt `  if you want to visualize program easily
- install with `conda env create -f environment.yml` if you want to create a separate python environment with all the dependencies

## How to use
### Core functions
Core function and representation are in the GPRL folder.  
```
.  
|── GPRL  
|   |── containers           # Fix a bug in qdpy grid 0.1.2.1 (current last stable version)
|   |── genetic_programming  # Individual definition of linear GP and Team for deap  
|   |── MCTS                 # Nested Monte-Carlo code  
|   |── utils                # Various utils and callback functions to run easily experiments  
|   |── algorithms.py        # deap like algorithm using toolbox  
|   |── factory.py           # Abstract class to make better used of toolbox between script  
|   |── UCB.py               # Subclass of deap base Fitness to use UCB  
└── ...
```
By using DEAP and these functions, we can conduct our experiments. Examples can be found at :  
<https://github.com/DEAP/deap/tree/master/examples>  

### Experiments script
Each experiment code is available  in separate script using DEAP. More details can be found in the `Readme.md` of experiments folder

### Main evolve script
The `evolve.py` script use configuration files in `.yml` to launch experiments. This script let you run QD, Tree GP and Linear GP.  
Basically, you can run an experiment with this command :
```
python evolve.py --conf /path/to/conf.yml
```
By default, the results is saved in the `results/` folder.  

### yaml configurations file
Here is a skeleton for the `conf.yml` file. This shows how an experiment can be set up
```
algorithm:
  name: # algorithm name in deap (algorithms.<name>) or algoritm name from GPRL (algo.name) 
  args:
    # args of the algorithm chosen (lambda_, mu, ngen ...)

population:
  init_size: #size of the population (int)
  
selection:
  name: # selection method for the evolutionnay algorithm. ex: selTournament (from deap.tools.sel*)
  args:
    # argument for the selection method. ex: tournsize: 5

individual: # Individual representation ("Tree" or "Linear")

params:
  env: # env-id from the gym/bullet env. ex:"MountainCarContinuous-v0"
  function_set: #Function set size ("small" or "extended")
  c: # Exploration constante for UCB (float)
  n_episodes: # Number of episode per evaluation (int)
  n_steps: # Number of step per evaluation (int)
  gamma: # Discount factor γ (float in [0,1])
  n_thread: # Number of thread to use (int)
  ... (many others depending of the individual representation (Tree or Linear). see conf/ for examples)
seed: #set seed for random
```

## See the result
Once an experiment is finished, you can see inspect results like in `tutorial.ipynb`. This notebook show how to see and run an individual from a saved population.

## Exemple of best policies found :
The notebook `best_policy.ipynb` shows best policy found by each method and demonstrate their portability and effeciency. A google colab version of this notebook can be found here: <https://colab.research.google.com/drive/11DdE4i2kY6dPXWtQ7Iwq4hejMXJ1XmNX?usp=sharing>

## Environments

\# | **Environment**       | **Name**                            |
--|-----------------------|-------------------------------------|
1 | Cartpole              | CartPole-v1                         |
2 | Acrobot               | Acrobot-v1                          |
3 | MountainCar           | MountainCarContinuous-v0            |
4 | Pendulum              | Pendulum-v0                         |
5 | InvDoublePend         | InvertedDoublePendulumBulletEnv-v0  |
6 | InvPendSwingUp        | InvertedPendulumSwingupBulletEnv-v0 |
7 | LunarLander           | LunarLanderContinuous-v2            |
8 | BipedalWalker         | BipedalWalker-v3                    |
9 | BipedalWalkerHardCore | BipedalWalkerHardcore-v3            |
10| Hopper                | HopperBulletEnv-v0                  |

`conf_gp#` and `conf_lingp#` are the configurations used for environment number listed above. (ex: `conf_gp124` is for environement Cartpole, Acrobot and Pendulum just modify `env` in params)

<!--- You can reproduce result for gp or lingp by running either `reproduce_results_gp.sh` or `reproduce_results_lingp.sh`. ---> 