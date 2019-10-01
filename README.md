# Evolutionary-computing

### TODO

- Check if STD of average life of population is correct in analysis line plot
- Implement second algorithm Mu, lambda. (First algorithm = Mu + lambda)
o Check with TA if 10 independent experiments should be ran for all enemies (30 experiments) for every algorithm
- Get the fitness for the best solutions and plot them in box plots (maybe get max-fitness over all generations)
- Statistical analysis to check if one algorithm is significantly better (checking arithmetic mean)
- Write paper

### evoman_framework directory

- The evoman directory

It contains all the basic files for the evoman framework and should not be changed

- specialist_A1 directory

CAN PROBABLY BE TRASHED BECAUSE IT DOES NOT CONTAIN LIFE OF THE AGENT YET

Contains all the results from the runs of the trained specialist model.

Generally, only the first three games are independently trained, there are two types of txt files,
results_best_[date]_[time]_enemy_[enemy_number] and results_log_[date]_[time]_enemy_[enemy_number]

The result best file contains the best weights found for the neural network during the whole training, the result log will list statistics for every generation and at the bottom of the file are the parameters used for training the model.


- specialist_A1_fixed directory

Same as specialist_A1 directory, but contains logs that contain life of agent

- specialist_agent_A1.py

Can be used to train a specialist agent for specified enemies

- specialist_agent_life.py

Fixed version of specialist_agent_A1.py, that logs the average population life

- specialist_agent_A1_demo.py

Can be used to visualize the results of weights found for the neural network. Specify the date_time combo at the top of the file for the matching results_best_ file and watch how it plays out by running it.