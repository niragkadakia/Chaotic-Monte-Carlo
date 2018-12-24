"""
Energy function classes.

Created by Nirag Kadakia at 12:00 06-09-2018
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp
import scipy.linalg as LA
from utils import diag_normal_pdf

class energy_func(object):
	"""
	TODO: fix so that if diagonal, no need to do dot. Do different ones for
			diagonal covariances and non-diagonal covariances. -- maybe not 
			needed, already overwrittten for quartic_separable
	"""
	
	
	def __init__(self, nD=2):
		self._dist_type = 'gaussian_std_diag'
		self._nD = nD
		self._diag = sp.ones(self.nD)
		
	@property
	def nD(self):
		return self._nD
	@property
	def dist_type(self):
		return self._dist_type
	@property
	def diag(self):
		return self._diag
	
	def f(self, x):
		return 0.5*sp.sum((x.T**2.0*self.diag).T, axis=0)
		
	def df(self, x):
		return (x.T*self.diag).T
		
	
class gaussian_diag(energy_func):
	"""
	Energy function for gaussian distribution with diag covariance.
	"""
	
	def __init__(self, nD=2, conditioning=0, diag=None):
		"""
		Args:
			nD (int): dimension of the sample space.
			conditioning (int); exponent of minimum inverse covariance; 
				inverse covariances are chosen log uniformly from 
				10^(-conditioning) to 10^0
			diag (list); if provided; manual list of inverse variances.
				Overrides `conditioning'.
		"""
		
		energy_func.__init__(self, nD)
		
		self._dist_type = "gaussian_diag"
		
		if diag is not None:
			assert len(diag) == self.nD, "'diag' must be a list or "\
				"numpy array of length nD (nD = %s; diag has length %s) "\
				% (self.nD, len(diag))
			self._diag = diag
		else:
			self._diag = 10.**sp.linspace(-conditioning, 0, self.nD)
		self.inv_cov = sp.diag(self._diag)
	
	def sample(self, num_samples):
		"""
		Sample from the distribution.
		"""
		
		std = 1./sp.sqrt(self.diag)
		return sp.random.normal(0, std, (num_samples, self.nD)).T
	
	
class gaussian_const_corr(energy_func):
	"""
	Energy function for gaussian with uniform covariance matrix:
	
	|1  p  p  p  p|    
	|p  1  p  p  p|
	|p  p  1  p  p|  +  noise 
	|p  p  p  1  p|
	|p  p  p  p  1|
	
	TODO: Allow for blocks
	"""
	
	def __init__(self, nD=2, seed=1, rho=0.2, epsilon=0.25, noiseD=5):
		""" 
		Args:
			nD (int): dimension of the sample space.
			seed (int): random number seed.
			rho (float): correlation matrix off-diag elements.
			epsilon (float): strength of noise; must be less than 1 - rho.
			noiseD (int): dimension of noise vectors u which generate noise.
				The higher the dimension, the larger spectrum diversity.
		"""
		
		energy_func.__init__(self, nD)
		
		assert epsilon < 1. - rho, "epsilon must be less than 1 - rho"
		
		self._dist_type = 'gaussian_const_corr'
		self._seed = int(seed)
		self._rho = rho
		self._epsilon = epsilon
		self._noiseD = int(noiseD)
		self.set_inv_cov()
		
	@property
	def seed(self):
		return self._seed
	@property
	def rho(self):
		return self._rho
	@property
	def epsilon(self):
		return self._epsilon
	@property
	def noiseD(self):
		return self._noiseD
		
	def set_inv_cov(self):
		"""
		Set the inverse correlation matrix.
		
		The correlation matrix is a sum of a constant and a noise matrix. 
		The noise matrix is built from dot products of random unit vectors 
		u_1,...,u_N each in R^M, entries ~ Uniform, s.t. |u| = 1. The noise 
		matrix is dot(u, u)*epsilon for non-diag entries; 0 on the diag. 
		The constant matrix has rho in the off-diag elements and 1 on 
		the diag.
		"""
		
		self.cov = sp.ones((self.nD, self.nD))*self.rho
		sp.fill_diagonal(self.cov, 1)
		
		sp.random.seed(self.seed)
		u_vecs = sp.random.uniform(-1, 1, size=(self.nD, self.noiseD))
		u_norms = sp.sqrt(sp.sum(u_vecs**2.0, axis=1))
		u_vecs = (u_vecs.T/u_norms).T
		noise_matrix = sp.dot(u_vecs, u_vecs.T)*self.epsilon
		sp.fill_diagonal(noise_matrix, 0)
		
		self.cov += noise_matrix
		self.inv_cov = LA.inv(self.cov)
		self._diag = sp.diag(self.inv_cov)
		
	def f(self, x):
		return 0.5*sp.sum(sp.dot(x.T, sp.dot(self.inv_cov, x)), axis=0)
	
	def df(self, x):
		return (sp.dot(x.T, self.inv_cov)).T
		
		
class gaussian_toeplitz_power(energy_func):
	"""
	Energy function for gaussian with toeplitz covariance matrix:
	
	|1    p^1  p^2  p^3  p^4|    
	|p^1  1    p^1  p^2  p^3|    
	|p^2  p^1  1    p^1  p^2|  +  noise
	|p^3  p^2  p^1  1    p^1|    
	|p^4  p^3  p^2  p^1  1  |    
	"""
	
	def __init__(self, nD=2, seed=1, rho=0.9, epsilon=0.03, noiseD=50):
		""" 
		Args:
			nD (int): dimension of the sample space.
			seed (int): random number seed.
			rho (float): correlation matrix off-diag elements.
			epsilon (float): strength of noise; must be less than 1 - rho.
			noiseD (int): dimension of noise vectors u which generate noise.
				The higher the dimension, the larger spectrum diversity.
		"""
		
		energy_func.__init__(self, nD)
		
		assert epsilon < (1. - rho)/(1. + rho), "epsilon must be "\
			   "less than (1-rho)/(1+rho)"
		
		self._dist_type = 'gaussian_toeplitz_power'
		self._seed = int(seed)
		self._rho = rho
		self._epsilon = epsilon
		self._noiseD = int(noiseD)
		self.set_inv_cov()
		
	@property
	def seed(self):
		return self._seed
	@property
	def rho(self):
		return self._rho
	@property
	def epsilon(self):
		return self._epsilon
	@property
	def noiseD(self):
		return self._noiseD
		
	def set_inv_cov(self):
		"""
		Set the inverse correlation matrix.
		
		The correlation matrix is a sum of a constant and a noise matrix. 
		The noise matrix is built from dot products of random unit vectors 
		u_1,...,u_N each in R^M, entries ~ Uniform, s.t. |u| = 1. The noise 
		matrix is dot(u, u)*epsilon for non-diag entries; 0 on the diag. 
		The constant matrix has rho in the off-diag elements and 1 on 
		the diag.
		"""
		
		cov_row = self.rho**sp.arange(self.nD)
		self.cov = LA.toeplitz(cov_row)
		sp.random.seed(self.seed)
		u_vecs = sp.random.uniform(-1, 1, size=(self.nD, self.noiseD))
		u_norms = sp.sqrt(sp.sum(u_vecs**2.0, axis=1))
		u_vecs = (u_vecs.T/u_norms).T
		noise_matrix = sp.dot(u_vecs, u_vecs.T)*self.epsilon
		sp.fill_diagonal(noise_matrix, 0)
		
		self.cov += noise_matrix
		self.inv_cov = LA.inv(self.cov)
		self._diag = sp.diag(self.inv_cov)
		
	def f(self, x):
		return 0.5*sp.sum(sp.dot(x.T, sp.dot(self.inv_cov, x)), axis=0)
	
	def df(self, x):
		return (sp.dot(x.T, self.inv_cov)).T

	
class quartic_separable(energy_func):
	"""
	"""
	
	def __init__(self, nD=2, diag=None):
		"""
		"""
		
		energy_func.__init__(self, nD)
		self._dist_type = 'quartic_separable'
		self.num_rej_sample_attempts = 2
		
		if diag is None:
			self._diag = sp.ones(self.nD)
		else:
			assert len(diag) == self.nD, \
				"length of diag passed to quartic_separable() "\
				"does not equal nD (%s)" % nD
			self._diag = diag
	
	def f(self, x):
		# TODO
	
		val = 0
		
		# diag quadratic elements
		val += sp.sum((x.T**2.0*self.diag).T, axis=0)
		
		# Quartic coupling terms, same scaling
		for iD in sp.arange(0, self.nD - 1, 2):
			val += x[iD]**2*x[iD + 1]**2*self.diag[iD]*self.diag[iD + 1]
			
		return val/2.0
		
	def df(self, x):
		# TODO
		
		grad = sp.zeros(x.shape)
		
		for iD in sp.arange(self.nD):
		
			# diag quadratic elements --> linear
			grad[iD] += x[iD]*self.diag[iD]
			
			# Quartic coupling terms --> linear*quadratic
			if iD % 2 == 0:
				if iD < self.nD - 1:
					grad[iD] += x[iD]*x[iD + 1]**2*self.diag[iD]*\
							self.diag[iD + 1]
			elif iD % 2 == 1:
				grad[iD] += x[iD - 1]**2*x[iD]*self.diag[iD - 1]*\
							self.diag[iD]
			
		return grad
		
	def sample(self, num_samples):
	
		half_nD = int(self.nD/2)
		
		# Number of candidates = #attempts per sample/walker for each var pair
		num_cands = num_samples*self.num_rej_sample_attempts*half_nD
		samples = sp.random.normal(size=(2, num_cands))
		
		# For each sample, proposal distriboution is 2D normal
		prop_maxs = (2*sp.pi)*diag_normal_pdf(samples, sp.ones(2), num_cands)
		prop_vals = sp.random.uniform(0, 1, num_cands)*prop_maxs
		test_vals = sp.exp(-(samples[0, :]**2.0 + samples[1, :]**2.0 + 
					samples[0, :]**2.0*samples[1, :]**2.0)/2.0)
		
		# Compare test against proposal values
		acc_array = test_vals > prop_vals
		acc_idxs = acc_array==1.
		
		# Get first N=num_samples non-rejected samples
		all_acc_samples = samples[:, acc_idxs]
		
		acc_samples = all_acc_samples[:, :num_samples*half_nD].\
						reshape(half_nD*2, -1, order='F')
		
		# Scale all by chaotic distribution diag scaling factor
		scaled_acc_samples = (acc_samples.T*(self.diag\
								[:half_nD*2]**-0.5)).T
		
		# Last element is from a normal distribution, scaled by covariance
		if self.nD % 2 == 1:
			final_p = sp.random.normal(0, self.diag[-1]**-0.5,
						size=num_samples)
			scaled_acc_samples = sp.vstack((scaled_acc_samples, final_p))
		
		#TODO: ensure enough walkers, or repeat.
		
		return scaled_acc_samples
		