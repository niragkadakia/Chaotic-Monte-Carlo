"""
Chaotic Monte Carlo class and associated methods


Created by Nirag Kadakia at 12:00 06-09-2018
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""


import numpy as np
from copy import deepcopy
import energy

		
class CHMC(object):
	"""
	TODO
	"""
	
	def __init__(self, nD=2, num_walkers=100, epsilon=0.25, 
				 num_steps_per_samp=1, num_iterations=1000, 
				 num_burnin=0):
	
		self._nD = nD
		
		self.num_walkers = num_walkers
		self.epsilon = epsilon
		self.num_steps_per_samp = num_steps_per_samp
		self.num_iterations = num_iterations
		self.num_burnin = num_burnin
		
		self._pX_type = None
		self._pP_type = None
		
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
			std = 1./np.sqrt(self.log_pX.diag)
			self.x_init = np.random.normal(0, std, 
			  (self.num_walkers, self.nD)).T
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
		
		for _ in range(self.num_steps_per_samp):
			x, p = self.leapfrog(x, p)
			
		return x, p
		
	def leap_prob(self, x_prop, p_prop):
		
		H_prop = self.H(x_prop, p_prop)
		dH = self.H_vec[-1] - H_prop
		if np.prod(np.isfinite(dH)) == 0:
			print ("Trajectory divergence...reduce epsilon")
			quit()
		acc_probs = np.ones(self.num_walkers)
		acc_probs[dH < 0] = np.exp(dH[dH < 0])
		
		return acc_probs

	def sample_x(self):
		return self.log_pX.sample(self.num_walkers)
		
	def sample_p(self):
		return self.log_pP.sample(self.num_walkers)
	
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
		rand_compare_vals = np.random.uniform(0, 1, (self.num_walkers))
		rej_idxs = np.where(acc_probs < rand_compare_vals)
		x_prop[:, rej_idxs] = self.x_vec[-1][:, rej_idxs]
		p_prop[:, rej_idxs] = self.p_vec[-1][:, rej_idxs]
		
		print (len(rej_idxs[0]))
		
		# Randomize momentum
		p_prop = self.sample_p()
		
		self.x_vec.append(x_prop)
		self.p_vec.append(p_prop)
		self.H_vec.append(self.H(x_prop, p_prop))
		
	def run(self):
		"""
		"""
		
		self.x_vec = None
		self.p_vec = None
		self.H_vec = None
		
		for iD in range(self.num_iterations):
			self.sample()
		self.x_vec = np.asarray(self.x_vec)
		self.p_vec = np.asarray(self.p_vec)
		
		