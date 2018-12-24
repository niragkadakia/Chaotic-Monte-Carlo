import scipy as sp
from collections import defaultdict
from scipy.stats import multivariate_normal
from utils import diag_normal_pdf
import time
import sys
sys.setrecursionlimit(5000)

### 	WRAPPER TO COUNT F AND DF EVALUATIONS

class counting_wrapper(object):
    def __init__(self, dist_X, dist_P):

        """
        A wrapper class for the energy function and gradient function, keeps
        track of how many times each one is called by the sampler, for later
        analysis.
        """
        
        self.Ex_external = dist_X.Ex.F
        self.Ep_external = dist_P.Ep.F
        self.dEdX_external = dist_X.Ex.dF
        self.dEdP_external = dist_P.Ep.dF

        self.Ex_count, self.Ep_count, self.dEdX_count, self.dEdP_count = 0,0,0,0
        self.ndims = len(dist_P.Ep.dF)


    def Ex(self, *args, **kwargs):
        self.Ex_count += 1
        return self.Ex_external

    def Ep(self, *args, **kwargs):
        self.Ep_count += 1
        return self.Ep_external

    def dEdX(self, *args, **kwargs):
        self.dEdX_count += 1/self.ndims
        return self.dEdX_external

    def dEdP(self, *args, **kwargs):
        self.dEdP_count += 1/self.ndims
        return self.dEdP_external


#### 		METHOD TO PERFORM HAMILTON AND P CORRUPTION STEPS

class sampler_obj():
	"""
	"""
	
	def __init__(self):
		self.dist_X 


class CHMC(object):
	def __init__(self, 
							 wrapper,
							 dist_X,
							 dist_P,
							 x_init,
							 epsilon = 0.1, 
							 alpha = 0.2, 
							 beta  = None,
							 seed = None,
							 num_leapfrog_steps = 10, 
							 num_look_ahead_steps = 4, 
							 display = 1,
							 p_corrupt_type = 'Monte_Carlo', 
							 p_corrupt_attempts = 50,
							 mix_up = 1,
							 args=(), 
							 kwargs={}):

		self.wrapper = wrapper
		self.dist_X = dist_X
		self.dist_P = dist_P
		self.args = args
		self.kwargs = kwargs
		self.display = display

		self.Ex_Type = self.dist_X.E_Type		
		self.Ep_Type = self.dist_P.E_Type
		self.coefficients_X = self.dist_X.x_coefficients
		self.coefficients_P = self.dist_P.p_coefficients
		
		self.M = num_leapfrog_steps
		self.K = num_look_ahead_steps

		self.p_corrupt_attempts = p_corrupt_attempts
		self.p_corrupt_type = p_corrupt_type
		self.last_accepted_P = []
		self.mix_up = mix_up

		self.ndims = x_init.shape[0]
		self.nbatch = x_init.shape[1]
		self.epsilon = epsilon
		self.beta = beta

		if beta is None:
			self.beta = alpha**(1./(self.epsilon*self.M))

		## Initialize the state at the first step of the iteration
		self.state = HMC_state(x_init.copy(), parent = self,seed=seed)
		if display > 1:
			print ("\nInitial X[0,:] is \n%s" % self.state.X[0,:])
			print ("Initial P[0,:] is \n%s" % self.state.P[0,:])

		## keep track of how many of each kind of transition happens
		self.counter = defaultdict(int)
		self.counter_steps = 0

		
	## Evaluate
	def Eval_Ex (self, X):
		exec(compile('self.tmp_val = eval(self.wrapper.Ex())','string','exec'))
		return self.tmp_val

	def Eval_Ep (self, X):	
		exec(compile('self.tmp_val = eval(self.wrapper.Ep())','string','exec'))
		return self.tmp_val
		
	def Eval_dEdX (self, X):
		tmp = []		
		for iN in sp.arange(self.ndims):
			exec(compile('self.tmp_val = eval(self.wrapper.dEdX()[iN])','string','exec'))
			tmp.append(self.tmp_val)
		return sp.asarray(tmp)

	def Eval_dEdP (self, X):
		tmp = []		
		for iN in sp.arange(self.ndims):
			exec(compile('self.tmp_val = eval(self.wrapper.dEdP()[iN])','string','exec'))
			tmp.append(self.tmp_val)
		return sp.asarray(tmp)

	def Eval_H (self, state):
		return state.Ex + state.Ep



	## Methods to sample P from the correct distribution

	def sample_P(self,P,args):

		attempts,p_corrupt_type = args
		dims_to_corrupt = len(P[:,0])
		batches_to_corrupt = len(P[0,:])

		def Rejection_Sampling(dims_to_corrupt=2, batches_to_corrupt=100,attempts=200,variance=sp.ones(2)):

			if self.Ep_Type == "Quartic_Conditioned_Separable":
				samples = sp.random.normal(size=(dims_to_corrupt,batches_to_corrupt*attempts))
				samples_pdf = diag_normal_pdf(samples,sp.ones(dims_to_corrupt),batches_to_corrupt*attempts)
				scale_factor = (2*sp.pi)**(dims_to_corrupt/2.)
				proposed_values = scale_factor*sp.random.uniform(0,1,size=(batches_to_corrupt*attempts))*samples_pdf
				acceptance_array = sp.exp(-(samples[0,:]**2.0 + samples[1,:]**2.0 + samples[0,:]**2.0*samples[1,:]**2.0)/2.0) > proposed_values
				accepted_samples = (samples*acceptance_array.T).T
				accepted_samples = accepted_samples[~(accepted_samples==0.).all(1)].T
				accepted_samples = accepted_samples[:,:batches_to_corrupt*dims_to_corrupt/2].T.reshape(-1,self.ndims)

			elif self.Ep_Type == "Quartic_Conditioned":
				samples = (sp.random.normal(size=(dims_to_corrupt,batches_to_corrupt*attempts)).T*variance**(-.5)).T
				samples_pdf = diag_normal_pdf(samples,variance**(-1),batches_to_corrupt*attempts)
				scale_factor = sp.prod(variance**(-.5))*(2*sp.pi)**(dims_to_corrupt/2.)
				proposed_values = scale_factor*sp.random.uniform(0,1,size=(batches_to_corrupt*attempts))*samples_pdf
				acceptance_array = sp.exp(-self.Eval_Ep(samples)) > proposed_values
				accepted_samples = (samples*acceptance_array.T).T
				accepted_samples = (accepted_samples[~(accepted_samples==0.).all(1)]).T

			return accepted_samples
			
		def Monte_Carlo(P):

			proposed_P = ((sp.random.normal(size = sp.shape(P)).T)*(sp.diag(self.coefficients_P)**(-0.5)/(self.ndims**(.5)))).T
			P_diff = -self.Eval_Ep(proposed_P + self.last_accepted_P[-1]) + self.Eval_Ep(self.last_accepted_P[-1])
			jump_probs = sp.ones(batches_to_corrupt)
			jump_probs[P_diff<0] = sp.exp(P_diff[P_diff<0])
			proposed_values = sp.random.uniform(0,1,size = batches_to_corrupt)
			acceptance_array = proposed_values < jump_probs
			accepted_samples = (proposed_P)*acceptance_array.T + self.last_accepted_P[-1]
			accepted_samples = accepted_samples*(sp.random.randint(0,2,size=sp.shape(P))*2-1)
			sp.random.shuffle(accepted_samples.T) # This may not be needed

			return accepted_samples



		if self.Ep_Type == "Quartic_Conditioned":			
			if p_corrupt_type == 'Rejection_Sampling':
				accepted_samples = Rejection_Sampling(self.ndims,batches_to_corrupt,attempts,sp.diag(self.coefficients_P))[:,:batches_to_corrupt]
			elif p_corrupt_type == 'Monte_Carlo':
				mix_up = sp.random.randint(0,self.mix_up)
				if mix_up == 1:
					print ("Random rejection sampling step")
					accepted_samples = Rejection_Sampling(self.ndims,batches_to_corrupt,attempts,sp.diag(self.coefficients_P))[:,:batches_to_corrupt]
				else:
					accepted_samples = Monte_Carlo(P)[:,:batches_to_corrupt]

		elif self.Ep_Type == "Quartic_Conditioned_Separable":			
			accepted_samples_tmp = Rejection_Sampling(dims_to_corrupt=2,batches_to_corrupt = batches_to_corrupt*self.ndims/2,attempts=2,variance = sp.ones(2))
			accepted_samples = (accepted_samples_tmp*sp.diag(self.coefficients_P)**(-.5)).T


		elif (self.Ep_Type == "Gaussian_Diag_Inv_Scaled") or (self.Ep_Type == "Gaussian_Diag_Custom_Scaled"):
			accepted_samples = (sp.random.normal(size=(dims_to_corrupt,batches_to_corrupt)).T*sp.diag(self.coefficients_P)**(-.5)).T


		self.last_accepted_P.append(accepted_samples)
		return accepted_samples[:,:batches_to_corrupt]



	## Method to perform 2nd order symplectic integration
	def leapfrog(self, state):
		idx = state.active_idx
		state.P[:,idx] += -self.epsilon/2. * state.dEdX[:,idx]
		state.update_dEdP()
		state.X[:,idx] += self.epsilon * state.dEdP[:,idx]
		state.update_dEdX()
		state.P[:,idx] += -self.epsilon/2. * state.dEdX[:,idx]
		return state


	## Method to perform leapfrom integration M times
	def L(self, state):
		for _ in range(self.M):
			state = self.leapfrog(state)
		state.update_Ep()
		state.update_Ex()
		return state

	## Method to determine probability of transition from Z1 to Z2
	def leap_prob(self, Z1, Z2):
		EZ1 = self.Eval_H(Z1)
		EZ2 = self.Eval_H(Z2)
		Ediff = EZ1 - EZ2
		p_acc = sp.ones((1, Ediff.shape[1]))
		p_acc[Ediff<0] = sp.exp(Ediff[Ediff<0])
		return p_acc

	def leap_prob_recurse(self, Z_chain, C, active_idx):
		"""
		Recursively compute to cumulative probability of transitioning from
		the beginning of the chain Z_chain to the end of the chain Z_chain.
		"""

		if sp.isfinite(C[0,-1,0]):
			## we've already visited this leaf
			cumu = C[0,-1,:].reshape((1,-1))
			return cumu, C

		if len(Z_chain) == 2:
			## the two states are one apart
			p_acc = self.leap_prob(Z_chain[0], Z_chain[1])
			p_acc = p_acc[:,active_idx]
			C[0,-1,:] = p_acc.ravel()
			return p_acc, C

		cum_forward, Cl = self.leap_prob_recurse(Z_chain[:-1], C[:-1,:-1,:], active_idx)
		C[:-1,:-1,:] = Cl
		cum_reverse, Cl = self.leap_prob_recurse(Z_chain[:0:-1], C[:0:-1,:0:-1,:], active_idx)
		C[:0:-1,:0:-1,:] = Cl

		H0 = self.Eval_H(Z_chain[0])
		H1 = self.Eval_H(Z_chain[-1])
		Ediff = H0 - H1
		Ediff = Ediff[:,active_idx]
		start_state_ratio = sp.exp(Ediff)
		prob =((sp.vstack  ((1. - cum_forward, start_state_ratio*(1. - cum_reverse)))).min(axis=0)).reshape((1,-1))
		cumu = cum_forward + prob
		C[0,-1,:] = cumu.ravel()
		return cumu, C


	def sampling_iteration(self):
		""" Perform a single sampling step. """

		# first do the HMC part of the step
		Z_chain = [self.state.copy(),]
		# use the same random number for comparison for the entire chain (for acceptances)
		rand_comparison = sp.random.rand(1, self.nbatch).ravel()
		# the current cumulative probability of acceptance
		p_cum = sp.zeros((1, self.nbatch))
		# the cumulative probability matrix, so we only need to visit each leaf once when recursing
		C = sp.ones((self.K+1, self.K+1, self.nbatch))*sp.nan
		# the current set of indices for samples that have not yet been accepted for a transition
		active_idx = sp.arange(self.nbatch, dtype=int)
		for kk in range(self.K):
			Z_chain.append(self.L(Z_chain[-1].copy()))
			# recursively calculate the cumulative probability of doing this many leaps
			p_cum, Cl = self.leap_prob_recurse(Z_chain, C[:kk+2, :kk+2, active_idx], active_idx)
			C[:kk+2, :kk+2, active_idx] = Cl
			# find all the samples that did this number of leaps, and update self.state with them
			accepted_idx = active_idx[p_cum.ravel() >= rand_comparison[active_idx]]
			self.counter['L%d'%(kk+1)] += len(accepted_idx)
			self.state.update(accepted_idx, Z_chain[-1])
			# update the set of active indices, so we don't do simulate trajectories for samples that are already assigned to a state
			active_idx = active_idx[p_cum.ravel() < rand_comparison[active_idx]]
			if len(active_idx) == 0:
				break
			Z_chain[-1].active_idx = active_idx
		# flip the momenutm for any samples that were unable to place elsewhere
		self.state.P[:,active_idx] = -self.state.P[:,active_idx]
		self.counter['F'] += len(active_idx)

		if self.display > 1:
			print "Transition counts ",
			for k in sorted(self.counter.keys()):
				print "%s:%d"%(k, self.counter[k]),

		# corrupt the momentum
		#TODO This formulation is invalid for CHMC 
		#transition = sp.random.randint(0,5)
		if (self.beta > 0):# and (transition == 1):
			self.state.P = self.state.P*sp.sqrt(1-self.beta) + self.sample_P(self.state.P,args=(self.p_corrupt_attempts,self.p_corrupt_type))*sp.sqrt(self.beta)	
		
		self.state.update_Ep()


	def sample(self, num_steps=100):
		"""
		Sample from the target distribution, for num_steps sampling steps.
		This is the function to call from external code.
		"""
		for iN in range(num_steps):
			if self.display > 1:
				print "sampling step %d / %d,"%(iN+1, num_steps),
			self.sampling_iteration()
			self.counter_steps += 1

		if self.display > 0:
			tot = 0
			for k in sorted(self.counter.keys()):
				tot += self.counter[k]
			print "Step %d, Transition fractions "%(self.counter_steps),
			for k in sorted(self.counter.keys()):
				print "%s:%g"%(k, self.counter[k]/float(tot)), 
			print()

		return (self.state.X.copy(), 
						self.state.P.copy(),
						self.state.Ex.copy(),
						self.state.Ep.copy())


#### 		CLASS TO HOLD THE STATE INFORMATION

class HMC_state(object):
	""" Holds all the state variables for HMC particles."""

	def __init__(self, X, parent, P=None, Ex=None, Ep=None, dEdX=None, dEdP=None,seed=None):
		"""
		Initialize HMC particle states.  Called by LAHMC class.
		Not user facing.
		"""
		self.parent = parent
		self.X = X
		self.P = P
		nbatch = X.shape[1]
		self.active_idx = sp.arange(nbatch)

		if P is None:
			self.ndims = self.X.shape[0]
			if seed is None:
				self.P = self.parent.sample_P(sp.zeros((self.ndims,nbatch)), args = (self.parent.p_corrupt_attempts,'Rejection_Sampling'))
			else:
				sp.random.seed(seed)
				self.P = self.parent.sample_P(sp.zeros((self.ndims,nbatch)), args = (self.parent.p_corrupt_attempts,'Rejection_Sampling'))
			self.parent.last_accepted_P.append(self.P)

		self.Ex = Ex
		self.Ep = Ep
		self.dEdX = dEdX
		self.dEdP = dEdP

		if Ex is None:
			self.Ex = sp.zeros((1,nbatch))
			self.update_Ex()
		if Ep is None:
			self.Ep = sp.zeros((1,nbatch))
			self.update_Ep()
		if dEdX is None:
			self.dEdX = sp.zeros(self.X.shape)
			self.update_dEdX()
		if dEdP is None:
			self.dEdP = sp.zeros(self.P.shape)
			self.update_dEdP()

	def update_Ex(self):
		self.Ex[:,self.active_idx] = self.parent.Eval_Ex(self.X[:,self.active_idx]).reshape((1,-1))
	def update_Ep(self):
		self.Ep[:,self.active_idx] = self.parent.Eval_Ep(self.P[:,self.active_idx]).reshape((1,-1))
	def update_dEdX(self):
		self.dEdX[:,self.active_idx] = self.parent.Eval_dEdX(self.X[:,self.active_idx])
	def update_dEdP(self):
		self.dEdP[:,self.active_idx] = self.parent.Eval_dEdP(self.P[:,self.active_idx])

	def copy(self):
		Z = HMC_state(X = self.X.copy(), 
								  P = self.P.copy(), 
								  Ex = self.Ex.copy(), 
									Ep = self.Ep.copy(), 
									dEdX = self.dEdX.copy(), 
									parent = self.parent)
		Z.active_idx = self.active_idx.copy()
		return Z

	def update(self, idx, Z):
		""" replace batch elements idx with state from Z """
		if len(idx)==0:
			return
		self.X[:,idx] = Z.X[:,idx]
		self.P[:,idx] = Z.P[:,idx]
		self.Ex[:,idx] = Z.Ex[:,idx]
		self.Ep[:,idx] = Z.Ep[:,idx]
		self.dEdX[:,idx] = Z.dEdX[:,idx]

