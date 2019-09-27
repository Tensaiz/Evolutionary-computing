###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# ASSIGNMENT 1 : Specialist agent                                             #
# Authors: Group 38:                                                          #
#   Mathijs Maijer                                                            #
#   Esra Solak                                                                #
#   Kasper Nicholas         			                                      #
###############################################################################

# Format: day-month-year_hours-minutes-seconds
FILE_DAY_TIME = '26-09-2019_00-57-14'

# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np


experiment_name = 'specialist_A1_demo'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 10

# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
				  playermode="ai",
				  player_controller=player_controller(n_hidden_neurons),
			  	  speed="normal",
				  enemymode="static",
				  level=2)

# tests saved demo solutions for each enemy
for en in range(1, 4):

    #Update the enemy
    env.update_parameter('enemies', [en])

    # Load specialist controller
    with open('specialist_A1/results_best_' + FILE_DAY_TIME + '_enemy_' + str(en) + '.txt') as f:
        sol = f.readline()
    sol = sol[1:-2].split(', ')
    sol = np.array(sol).astype(np.float)
    print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY '+str(en)+' \n')
    print(env.play(sol))
