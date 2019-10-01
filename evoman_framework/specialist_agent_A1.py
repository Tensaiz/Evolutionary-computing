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
from datetime import datetime
import random
import numpy as np
from math import fabs,sqrt
import glob

# EC framework
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

### Configuration

experiment_name = 'specialist_A1'
algorithm_name = 'Mu + Lambda'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  logs="off",
                  speed="fastest")

n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

domain_upper = 1
domain_lower = -1
n_pop = 100
n_gens = 30 # 30
mutation_p = 0.2
cross_p = 0.5
mu = 100
lambda_ = 200

# Evo framework config

# runs simulation
def simulation(individual):
    f,p,e,t = env.play(pcont=np.array(individual))
    return f

def evaluation(individual):
    return (simulation(individual),)

# Initialize the fitness and individual classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_weight", random.uniform, domain_lower, domain_upper)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_weight, n_vars)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluation)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

name_suffix = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')

for en in range(1, 4):
    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    #Update the enemy
    env.update_parameter('enemies', [en])

    print('\n Evolving specialist on enemy: '+str(en)+' \n')

    # pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cross_p, mutpb=mutation_p, ngen=n_gens, stats=stats, halloffame=hof, verbose=True)
    pop, log = algorithms.eaMuCommaLambda(pop, toolbox, mu=mu, lambda_=lambda_, cxpb=cross_p, mutpb=mutation_p, ngen=n_gens, stats=stats, halloffame=hof, verbose=True)

    # saves results for first pop
    f  = open(experiment_name+'/results_log_' + name_suffix + '_enemy_' + str(en) + '.txt','a')
    f.write(str(log))
    f.write(
        '\n' +
        'n_hidden_neurons = ' + str(n_hidden_neurons) + '\n'
        'domain_upper = ' + str(domain_upper) + '\n'
        'domain_lower = ' + str(domain_lower) + '\n'
        'n_pop = ' + str(n_pop) + '\n'
        'n_gens = ' + str(n_gens) + '\n'
        'mutation_p = ' + str(mutation_p) + '\n'
        'cross_p = ' + str(cross_p) + '\n'
        'Mu = ' + str(mu) + '\n'
        'Lambda = ' + str(lambda_) + '\n'
        'Algorithm = ' + algorithm_name + '\n'
    )
    f.close()

    f  = open(experiment_name+'/results_best_' + name_suffix + '_enemy_' + str(en) + '.txt','a')
    f.write(str(hof[0]))
    f.close()
