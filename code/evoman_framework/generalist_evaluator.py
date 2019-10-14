###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# ASSIGNMENT 2 : Generalist agent evaluator                                   #
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
import numpy as np

experiment_name = 'generalist_eval'

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

# Get the best solutions
solutions = os.listdir('./generalist_A2')
solutions = list(filter(lambda x: 'gen_best' in x, solutions))

fitness = []
victory = []

for solution in solutions:
    enemies = solution.split('_')[-1][0:-4]
    enemies = enemies.strip('][').split(', ')
    enemies = list(map(lambda x: int(x), enemies))

    print('\n TESTING GENERALIST SOLUTION TRAINED ON ENEMIES: ' + str(enemies) + ' \n')

    fitness.append([])
    victory.append([])

    # Test solutions for every enemy
    for en in range(1, 9):
        #Update the enemy
        env.update_parameter('enemies', [en])

        # Load specialist controller
        with open('./generalist_A2/' + solution) as f:
            sol = f.readline()
        sol = sol[1:-2].split(', ')
        sol = np.array(sol).astype(np.float)

        # Play game and get stats
        f, p, e, t = env.play(sol)
        fitness[-1].append(f)
        if p > 0:
            victory[-1].append(1)
        else:
            victory[-1].append(0)
    print('PERFORMANCE OF:')
    print(solution)
    print('\n' + 'Fitnesses:\n' + str(fitness[-1]) + '\n' + 'Victories:' + '\n' + str(victory[-1]))

wins = list(map(lambda x: x.count(1), victory))
most_wins = max(wins)
index = wins.index(most_wins)

print('Most wins: ' + str(most_wins) + ' achieved by solution: \n' + solutions[index])