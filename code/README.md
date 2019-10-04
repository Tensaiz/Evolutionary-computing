# Evolutionary-computing

### Main files

#### 38.pdf

The paper written for this assignment

#### Analysis.ipynb

The python notebook that was used to analyse the data and gather all the results/images used in the paper

#### install.sh

Shell script that can be used to install the required packages and repositories

### evoman_framework directory

- The evoman directory

It contains all the basic files for the evoman framework and should not be changed

- specialist_A1_fixed directory

Contains all the results from the runs of the trained specialist model with the mu + lambda strategy.

Generally, only the first three games are independently trained, there are two types of txt files,
results_best_[date]_[time]_enemy_[enemy_number] and results_log_[date]_[time]_enemy_[enemy_number]

The result best file contains the best weights found for the neural network during the whole training, the result log will list statistics for every generation and at the bottom of the file are the parameters used for training the model.

- specialist_A2_fixed directory

Same as specialist_A1_fixed directory, but contains results for the mu,lambda strategy.

- specialist_agent_life.py

Can be used to train a specialist agent for specified enemies

- specialist_agent_A1_demo.py

Can be used to visualize the results of weights found for the neural network. Specify the date_time combo at the top of the file for the matching results_best_ file and watch how it plays out by running it.

- demo_controller.py

Contains the code for the player controller ANN, that maps the weights found during training to player actions.