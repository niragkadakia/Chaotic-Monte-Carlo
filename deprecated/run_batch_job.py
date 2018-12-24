"""
Run batch job CHMC run.

Created by Nirag Kadakia at 22:50 04-17-2018
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license,
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp
import sys
sys.path.append('../src/')
from sampler import run_sampling
from sampler import load_results
from plot import plot_covariance
from plot import calc_covariance_errors
from plot import plot_correlation


if len(sys.argv) != 1:
	iter_min = int(sys.argv[1])-1
	iter_max = iter_min + 1
	e_min = float(sys.argv[2])
	e_max = e_min
	e_step = e_min
	beta = int(sys.argv[3])
else:
	iter_min = 0
	iter_max = 1
	e_min, e_max, e_step = .1,.1,.1
	beta = 0

### User-defined information about the annealing procedures

L = 50
ndims = 20
num_steps = 100
nbatch = 100
sampler_list = ('CHMC','HMC')
epsilons = sp.arange(e_min,e_max+e_step,e_step)
# = 0,1,2 for more information being displayed during sampling
display=1
# Types: Quartic_Conditioned_Separable, Quartic_Conditioned
CHMC_Type = 'Quartic_Conditioned_Separable' 
# Types: Gaussian_Diag or Gaussian_Non_Diag
Ex_Type = "Gaussian_Non_Diag"  
# Below is only relevant for "Gaussian_Non_Diag" Ex_Types. Types: Uniform, Toeplitz_Exponential, Toeplitz_Uniform
Ex_Structure = "Uniform" 
# Hard limits on values in uniform covariance matrices when searching for positive definite matrices
cov_lims = [-0.13,0.13]
# P is resampled either via rejection sampling or Monte Carlo + mix_up methods
p_corrupt_type = 'Rejection_Sampling' #'Monte_Carlo'


## Indicate what plots to save and data to generate
plot_errors = True
plot_corr = False

iE_counter = 0
for iE in epsilons:

	print ("\n\n\n\n\n\########################\n\n     iE   =   %s         \
         \n\n#######################\n\n\n\n\n\n" % iE)
	iN_counter = 0
	errors = sp.zeros((num_steps,len(sampler_list), 2))

	for iN in sp.arange(iter_min, iter_max):

		print ("..........................................     iN   =   %s" % iN)
		print ("\n beta = %s" % beta)
		filename,dist_X = run_sampling(
			base_filename = "Test", 
			num_steps = num_steps, 
			nbatch = nbatch,
			seed= iN, 
			ndims = ndims,
			epsilon = iE,
			L = L,
			display=display,
			log_conditioning_X = 0,
			log_conditioning_P = 0,
			Ex_Type = Ex_Type,
			Ex_Structure = Ex_Structure,
			cov_lims = cov_lims,   
			sampler_list = sampler_list,
			CHMC_p_corrupt_type = p_corrupt_type,
			CHMC_Type = CHMC_Type,
			beta = beta
			)

		history = load_results(filename)

		if plot_errors == True:
	
			errors, samp_names = calc_covariance_errors(history,dist_X)

			sp.savetxt('Data/Errors/On_Diag,D=%s,E=%s,L=%s,Batch=%s,N=%s,beta=%s,iN=%s.dat' % (ndims,epsilons[iE_counter],L,nbatch,num_steps,beta,iN), errors[:,:,0],fmt  = '%.3e',delimiter='\t')
			sp.savetxt('Data/Errors/Off_Diag,D=%s,E=%s,L=%s,Batch=%s,N=%s,beta=%s,iN=%s.dat' % (ndims,epsilons[iE_counter],L,nbatch,num_steps,beta,iN), errors[:,:,1],fmt  = '%.3e',delimiter='\t')

		if plot_corr == True:
			corr = plot_correlation(history)
			sp.savetxt('Data/Correlations/D=%s,E=%s,L=%s,Batch=%s,N=%s,beta=%s,iN=%s.dat' %(ndims,epsilons[iE_counter],L,nbatch,num_steps,beta,iN), sp.asarray(corr).T, fmt = '%.3e', delimiter = '\t')

		
		iN_counter +=1


	iE_counter += 1

