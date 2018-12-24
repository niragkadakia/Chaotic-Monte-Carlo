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
sys.path.append('../src/')
from CHMC import CHMC
import scipy as sp
import scipy.linalg as LA
import os
from local_vars import data_dir
import pickle

DATA_DIR = data_dir()

init_seed = int(sys.argv[1])

a = CHMC(nWalkers=1000, nD=50, epsilon=0.2, nSteps_per_samp=10)
a.set_pX_dist(pX_type='gaussian_toeplitz_power')
a.set_pP_dist(pP_type='gaussian_diag', diag=a.log_pX.diag**-1.0)
#a.set_pP_dist(pP_type='quartic_separable', diag=a.log_pX.diag**-1.0)
sp.random.seed(init_seed)
a.set_init_state()
a.p_init *= 2 ### NEED TO GET INITIAL sampled outside region

steps = 30
burnin = 3

for iD in range(steps):
	a.sample()
a.x_vec = sp.asarray(a.x_vec)
a.p_vec = sp.asarray(a.p_vec)

dir = '%s/errs/%s/%s/%.3f' % (DATA_DIR, a.pX_type, a.pP_type, a.epsilon)
if not os.path.exists(dir):
	os.makedirs(dir)
filename = '%s/%s.pkl' % (dir, init_seed)
with open(filename, 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)