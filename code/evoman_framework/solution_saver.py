import numpy as np

# Config
OUT_FOLDER = './numpy_solutions/'
FILE = 'gen_best_14-10-2019_00-55-57_enemies_[7, 8].txt'
OUT_NAME = '38.txt'

# Read in custom solution file
with open('./generalist_A2/' + FILE) as f:
    sol = f.readline()
sol = sol[1:-2].split(', ')
sol = np.array(sol).astype(np.float)

# Save as numpy array
np.savetxt(OUT_FOLDER + OUT_NAME, sol)