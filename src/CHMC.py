"""
CHMC class and sampler class


Created by Nirag Kadakia at 12:00 06-09-2018
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""


import scipy as sp
from copy import deepcopy
import energy

		
class CHMC(object):
	"""
	"""
	
	def __init__(self, nD=2, nWalkers=100, epsilon=0.25,
					nSteps_per_samp=1):
	
		self._nD = nD
		
		self.nWalkers = nWalkers
		self.epsilon = epsilon
		self.nSteps_per_samp = nSteps_per_samp
		
		self._pX_type = None
		self._pP_type = None
		
		self.x_vec = None
		self.p_vec = None
		self.H_vec = None
		
		self.Ex_count = 0
		self.Ep_count = 0
		self.dEx_count = 0
		self.dEp_count = 0

	@property
	def nD(self):
		return self._nD
	@property
	def pX_type(self):
		return self._pX_type
	@property
	def pP_type(self):
		return self._pP_type
	
		
	def set_pX_dist(self, pX_type='gaussian_diag', **kwargs):
		
		assert hasattr(energy, '%s' % pX_type), \
			'Model class "%s" not in energy module' % pX_type
		exec ('self.log_pX = energy.%s(self.nD, **kwargs)' % pX_type)
		
		self._pX_type = pX_type
	
	def set_pP_dist(self, pP_type='gaussian_diag', **kwargs):
		
		assert hasattr(energy, '%s' % pP_type), \
			'Model class "%s" not in energy module' % pP_type
		exec ('self.log_pP = energy.%s(self.nD, **kwargs)' % pP_type)
		
		self._pP_type = pP_type
	
	def set_init_state(self):
		if hasattr(self.log_pX, 'sample'):
			self.x_init = self.sample_x()
		else:
			std = 1./sp.sqrt(self.log_pX.diag)
			self.x_init = sp.random.normal(0, std, (self.nWalkers, self.nD)).T
	 	self.p_init = self.sample_p()
			
	def Ex(self, x):
		return self.log_pX.f(x)
	
	def Ep(self, p):
		return self.log_pP.f(p)
	
	def dEx(self, x):
		return self.log_pX.df(x)

	def dEp(self, p):
		return self.log_pP.df(p)

	def H(self, x, p):
		return self.Ex(x) + self.Ep(p)
	
	def leapfrog(self, x, p):
		
		p += -self.epsilon/2.*self.dEx(x)
		x += self.epsilon*self.dEp(p)
		p += -self.epsilon/2.*self.dEx(x)
		
		return x, p

	def leapfrog_chain(self, x, p):
		
		for _ in range(self.nSteps_per_samp):
			x, p = self.leapfrog(x, p)
			
		return x, p
		
	def leap_prob(self, x_prop, p_prop):
		
		H_prop = self.H(x_prop, p_prop)
		dH = self.H_vec[-1] - H_prop
		if sp.prod(sp.isfinite(dH)) == 0:
			print "Trajectory divergence...reduce epsilon" 
			quit()
		acc_probs = sp.ones(self.nWalkers)
		acc_probs[dH < 0] = sp.exp(dH[dH < 0])
		
		return acc_probs

	def sample_x(self):
		return self.log_pX.sample(self.nWalkers)
		
	def sample_p(self):
		return self.log_pP.sample(self.nWalkers)
	
	def sample(self):
	
		assert hasattr(self, 'log_pX'), 'Before sampling, first set the '\
			'target distribution via set_pX_dist()'
		assert hasattr(self, 'log_pP'), 'Before sampling, first set the '\
			'momentum distribution via set_pP_dist()'
		assert hasattr(self, 'x_init'), 'Before sampling, first set the '\
			'initial state via set_init_state()'
		
		# TODO: make this all numpy vectors rather than appending to lists
		if self.x_vec is None:
			self.x_vec = [self.x_init]
			self.p_vec = [self.p_init]
			self.H_vec = [self.H(self.x_init, self.p_init)]
			
		x_prop = deepcopy(self.x_vec[-1])
		p_prop = deepcopy(self.p_vec[-1])
		x_prop, p_prop = self.leapfrog_chain(x_prop, p_prop)
		
		# Calculate acceptance probability and accept new samples or repeat old
		acc_probs = self.leap_prob(x_prop, p_prop)
		rand_compare_vals = sp.random.uniform(0, 1, (self.nWalkers))
		rej_idxs = sp.where(acc_probs < rand_compare_vals)
		x_prop[:, rej_idxs] = self.x_vec[-1][:, rej_idxs]
		p_prop[:, rej_idxs] = self.p_vec[-1][:, rej_idxs]
		
		print len(rej_idxs[0])
		
		# Randomize momentum
		p_prop = self.sample_p()
		
		self.x_vec.append(x_prop)
		self.p_vec.append(p_prop)
		self.H_vec.append(self.H(x_prop, p_prop))
		
		
		
		
		
def integrator():
	errs = []
	for iR in range(1):
		nD = 50
		a = CHMC(nWalkers=1000, nD=nD, epsilon=0.15, nSteps_per_samp=50)
		a.set_pX_dist(pX_type='gaussian_toeplitz_power')
		a.set_pP_dist(pP_type='gaussian_diag', diag=a.log_pX.diag**-1.0)
		#a.set_pP_dist(pP_type='quartic_separable', diag=a.log_pX.diag**-1.0)
		sp.random.seed()
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
		
		steps = 2000
		burnin = 1000
		
		for iD in range(steps):
			print (iD)
			a.sample()
		a.x_vec = sp.asarray(a.x_vec)
		a.p_vec = sp.asarray(a.p_vec)
		
		vals = sp.reshape(sp.rollaxis(a.x_vec[burnin:], -1), ((-1, a.nD))).T
		cov = sp.cov(vals)
		
		#import matplotlib.pyplot as plt
		#from mpl_toolkits.mplot3d import Axes3D
		import scipy.linalg as LA
		#plt.subplot(121)
		#plt.imshow(LA.inv(cov))
		#plt.subplot(122)
		#plt.imshow(a.log_pX.inv_cov)
		#plt.show()
		err = (a.log_pX.inv_cov - LA.inv(cov))**2.0
		for iN in range(nD):
			err[iN, iN] = 0
		print (sp.sum(err**2.0)**0.5)
		errs.append(sp.sum(err**2.0)**0.5)
		#plt.imshow(err)
		#plt.show()
		#fig = plt.figure()
		#ax = fig.add_subplot(111, projection='3d')
		#plt.plot(a.x_vec[:, 0, 0], a.x_vec[:, -1, 0])
		#plt.show()
		
	print (errs)
	print (sp.mean(errs))