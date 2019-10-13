###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# ASSIGNMENT 2 : Generalist agent                                             #
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
import itertools
from math import fabs,sqrt
import glob

# EC framework
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

### ALGORITHMS ###

def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    life_mean = 0

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'life_avg'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    eval = list(toolbox.map(toolbox.evaluate, invalid_ind))

    fitnesses = [e[0] for e in eval]
    life_list = [e[1] for e in eval]
    life_mean = np.mean(life_list)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), life_avg=life_mean, **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        eval = list(toolbox.map(toolbox.evaluate, invalid_ind))
        fitnesses = [e[0] for e in eval]
        life_list = [e[1] for e in eval]
        life_mean = np.mean(life_list)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), life_avg=life_mean, **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

def eaMuCommaLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                    stats=None, halloffame=None, verbose=__debug__):

    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    life_mean = 0

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'life_avg'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    eval = list(toolbox.map(toolbox.evaluate, invalid_ind))

    fitnesses = [e[0] for e in eval]
    life_list = [e[1] for e in eval]
    life_mean = np.mean(life_list)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), life_avg=life_mean, **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        eval = list(toolbox.map(toolbox.evaluate, invalid_ind))
        fitnesses = [e[0] for e in eval]
        life_list = [e[1] for e in eval]
        life_mean = np.mean(life_list)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), life_avg=life_mean, **record)
        if verbose:
            print(logbook.stream)
    return population, logbook

##################


### Configuration
experiment_name = 'generalist_A2'
algorithm_name = 'Mu + Lambda'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

domain_upper = 1
domain_lower = -1
n_pop = 100
n_gens = 30
mutation_p = 0.2
cross_p = 0.5
mu = 100
lambda_ = 200

# Initialise enemy combinations
# all_enemies = list(range(1, 8+1))
# num_enemies = 2
# all_combos = list(itertools.combinations(all_enemies, num_enemies))

all_combos = [[2,4], [2,6], [7, 8]]
# all_combos = [[1, 5, 6], [1, 2, 5], [2, 5, 6]]

# runs simulation
def simulation(individual):
    f, p, e, t = env.play(pcont=np.array(individual))
    return f, p

def evaluation(individual):
    f, p = simulation(individual)
    return [(f,), p]

name_suffix = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')

for en in all_combos:
    enemies = list(en)
    # Evo framework config
    # initializes simulation in multi evolution mode, for two static enemies.
    env = Environment(experiment_name=experiment_name,
                      enemies=enemies,
                      multiplemode="yes",
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      logs="off",
                      speed="fastest")

    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

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

    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    print('\n Evolving generalist on enemy combination: ' + str(en) + ' \n')

    # pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cross_p, mutpb=mutation_p, ngen=n_gens, stats=stats, halloffame=hof, verbose=True)
    pop, log = eaMuPlusLambda(pop, toolbox, mu=mu, lambda_=lambda_, cxpb=cross_p, mutpb=mutation_p, ngen=n_gens, stats=stats, halloffame=hof, verbose=True)

    # saves results for first pop
    f = open(experiment_name+'/gen_' + name_suffix + '_enemies_' + str(enemies) + '.txt', 'a')
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
        'Enemies = ' + str(enemies) + '\n'
    )
    f.close()

    f = open(experiment_name+'/gen_best_' + name_suffix + '_enemies_' + str(enemies) + '.txt','a')
    f.write(str(hof[0]))
    f.close()
