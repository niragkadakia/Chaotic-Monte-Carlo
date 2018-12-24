import sys
sys.path.append('../src/')
from CHMC_test import CHMC
import scipy as sp
import scipy.linalg as LA

data_dir = '../data'

for init_seed in range(0, 10):
	#init_seed = int(sys.argv[1])

	errs = []
	nD = 50
	a = CHMC(nWalkers=1000, nD=nD, epsilon=0.2, nSteps_per_samp=10)
	a.set_pX_dist(pX_type='gaussian_toeplitz_power')
	a.set_pP_dist(pP_type='gaussian_diag', diag=a.log_pX.diag**-1.0)
	#a.set_pP_dist(pP_type='quartic_separable', diag=a.log_pX.diag**-1.0)
	sp.random.seed(init_seed)
	a.set_init_state()
	a.p_init *= 2 ### NEED TO GET INITIAL sampled outside region

	"""
	from scipy.optimize import check_grad
	for val in range(10):
		i = sp.random.normal(0, 20, 4)
		suc = check_grad(a.log_pP.f, a.log_pP.df, i)
		print (suc)
	quit()
	"""

	steps = 10000
	burnin = 500

	for iD in range(steps):
		a.sample()
	a.x_vec = sp.asarray(a.x_vec)
	a.p_vec = sp.asarray(a.p_vec)

	vals = sp.reshape(sp.rollaxis(a.x_vec[burnin:], -1), ((-1, a.nD))).T
	cov = sp.cov(vals)

	err = (a.log_pX.inv_cov - LA.inv(cov))**2.0
	for iN in range(nD):
		err[iN, iN] = 0
	mse = sp.sum(err*2.0)**0.5
	#sp.savetxt('%s/%s.txt' % (data_dir, init_seed), err, fmt='%.4f')
	sp.savetxt('%s/%s/%s.txt' % (data_dir, a.pP_type, init_seed), [mse], fmt='%.4f')