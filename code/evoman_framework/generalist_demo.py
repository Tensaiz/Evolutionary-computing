###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# ASSIGNMENT 2 : Generalist agent                                             #
# Authors: Group 38:                                                          #
#   Mathijs Maijer                                                            #
#   Esra Solak                                                                #
#   Kasper Nicholas         			                                      #
###############################################################################

# imports framework
import sys,os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

import numpy as np

from types import MethodType


FOLDER = './numpy_solutions/'
FILE = '38.txt'

n_hidden_neurons = 10

# initializes environment for multi objetive mode (generalist)  with static enemy and ai player

experiment_name = 'generalist_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(experiment_name=experiment_name,
				  playermode="ai",
				  player_controller=player_controller(n_hidden_neurons),
		  		  speed="normal",
				  enemymode="static",
				  level=2)

sol = np.loadtxt(FOLDER + FILE)
print('\n LOADING SAVED GENERALIST SOLUTION FOR ALL ENEMIES \n')

# tests saved demo solutions for each enemy
for en in range(1, 9):

	#Update the enemy
	env.update_parameter('enemies',[en])

	f, p, e, t = env.play(sol)


print('\n  \n')