###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# ASSIGNMENT 1 : Specialist agent                                             #
# Authors: Group 38:                                                          #
#   Mathijs Maijer                                                            #
#   Esra Solak                                                                #
#   Kasper Nicholas         			                                      #
###############################################################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob

experiment_name = 'specialist_A1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 5

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

dom_u = 1
dom_l = -1
npop = 5
gens = 30
mutation = 0.2
last_best = 0

weights = np.random.uniform(dom_l, dom_u, (npop, n_vars))

list(map(lambda y: env.play(pcont=y), weights))
