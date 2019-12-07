
"""
Run a Monte Carlo sampling and save data to file

Created by Nirag Kadakia at 12:00 12-23-2018
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import sys
import numpy as np
import scipy.linalg as LA
import os
import pickle
import matplotlib.pyplot as plt
sys.path.append('../src/')
from CHMC import CHMC
from local_vars import data_dir

DATA_DIR = data_dir()
init_seed = int(sys.argv[1])

epsilons['gaussian_diag'] = np.arange(0.1, 1.7, 0.05)
epsilons['quartic_separable'] = np.arange(0.1, 0.3, 0.005)

for pP_type in ['gaussian_diag', 'quartic_separable']:
	for epsilon in epsilons[pP_type]:
		
		a = CHMC(num_walkers=10, nD=50, epsilon=epsilon, 
				 num_steps_per_samp=50, num_iterations=1000, 
				 num_burnin=0)
		a.set_pX_dist(pX_type='gaussian_const_corr')
		a.set_pP_dist(pP_type=pP_type, diag=a.log_pX.diag**-1.0)
		np.random.seed(init_seed)
		a.set_init_state()
		a.p_init *= 1.5 
		
		a.run()
		
		## TODO , calculate errors
		
		dir = '%s/errs/%s/%s/%.3f' % (DATA_DIR, a.pX_type, 
									  a.pP_type, a.epsilon)
		if not os.path.exists(dir):
			os.makedirs(dir)
		filename = '%s/%s.pkl' % (dir, init_seed)
		with open(filename, 'wb') as handle:
			pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
