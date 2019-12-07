
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
import numpy as np
import scipy.linalg as LA
import os
from local_vars import data_dir
import pickle

DATA_DIR = data_dir()

init_seed = int(sys.argv[1])

for epsilon in np.arange(0.1, 1.7, 0.05):
#for epsilon in np.arange(0.1, 0.3, 0.005):
	a = CHMC(num_walkers=10, nD=50, epsilon=epsilon, nSteps_per_samp=50)
	#a.set_pX_dist(pX_type='gaussian_toeplitz_power')
	a.set_pX_dist(pX_type='gaussian_const_corr')
	#a.set_pX_dist(pX_type='gaussian_diag', conditioning=2)
	a.set_pP_dist(pP_type='gaussian_diag', diag=a.log_pX.diag**-1.0)
	#a.set_pP_dist(pP_type='quartic_separable', diag=a.log_pX.diag**-1.0)
	np.random.seed(init_seed)
	a.set_init_state()
	a.p_init *= 1.5 ### NEED TO GET INITIAL sampled outside region

	steps = 100
	burnin = 0

	for iD in range(steps):
		a.sample()
	a.x_vec = np.asarray(a.x_vec)
	a.p_vec = np.asarray(a.p_vec)
	import scipy.linalg as LA
	import matplotlib.pyplot as plt

	data = np.zeros((a.nD, (steps - burnin + 1)*a.num_walkers))
	for iD in range(a.nD):
		data[iD] = (a.x_vec[burnin:, iD, :]).flatten()
	cov = np.cov(data)

	#plt.plot(a.x_vec[:, 0, 0], a.x_vec[:, 1, 0])
	#plt.show()
	#err = (a.log_pX.cov - cov)**2.0
	err = (LA.inv(a.log_pX.inv_cov) - cov)**2.0
	diagz = np.diag(np.diag(err))
	#plt.subplot(211)
	#plt.imshow(LA.inv(a.log_pX.inv_cov))
	#plt.subplot(212)
	#plt.imshow(cov)
	#plt.show()
	print (epsilon, np.mean(err - diagz))
	plt.scatter(epsilon, np.mean(err - diagz))
plt.yscale('log')
plt.show()
dir = '%s/errs/%s/%s/%.3f' % (DATA_DIR, a.pX_type, a.pP_type, a.epsilon)
if not os.path.exists(dir):
	os.makedirs(dir)
filename = '%s/%s.pkl' % (dir, init_seed)
with open(filename, 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)