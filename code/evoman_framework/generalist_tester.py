###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# ASSIGNMENT 2 : Generalist agent tester                                      #
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
from datetime import datetime
import numpy as np
import pandas as pd

experiment_name = 'generalist_test'
name_suffix = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 10

# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
				  playermode="ai",
				  player_controller=player_controller(n_hidden_neurons),
			  	  speed="fastest",
				  enemymode="static",
				  level=2)


# Get solutions and their enemies
solutions = []
enemies = []

folder = './generalist_A2/'

# Get the best solutions
solutions = os.listdir(folder)
solutions = list(filter(lambda x: 'gen_best' in x, solutions))
solutions = list(filter(lambda x: '[2, 4]' in x, solutions))
gain_complete = []

# Check for all solutions
for solution in solutions:
    enemies = solution.split('_')[-1][0:-4]
    enemies = enemies.strip('][').split(', ')
    enemies = list(map(lambda x: int(x), enemies))
    gain_sol = []

    # Repeat each solution 5 times
    for i in range(1, 6):
        gain_run = []

        # Test solutions for every enemy
        for en in range(1, 9):
            #Update the enemy
            env.update_parameter('enemies', [en])

            # Load specialist controller
            with open(folder + solution) as f:
                sol = f.readline()
            sol = sol[1:-2].split(', ')
            sol = np.array(sol).astype(np.float)

            # Play game and get stats
            f, p, e, t = env.play(sol)

            # Append gain of 1 enemy to the list of run
            gain_run.append(p-e)

        # Append list of gains for 8 enemies
        gain_sol.append(gain_run)

    df = pd.DataFrame(gain_sol)
    columns = ["en_1", "en_2", "en_3", "en_4", "en_5", "en_6", "en_7", "en_8"]
    df.columns = columns
    df.to_csv('/generalist_A2_tester/generalist_test_' + str(solution) + '.csv')


