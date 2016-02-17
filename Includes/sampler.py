import math
import scipy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Includes.energy import Kinetic_Energy
from Includes.energy import Potential_Energy
from Includes.CHMC import counting_wrapper
from Includes.CHMC import CHMC
from collections import OrderedDict
import time


def save_results(history, filename):
	sp.savez(filename, history=history)


def load_results(filename):
	data = sp.load(filename)
	history = data['history'][()]
	return history    


def run_sampling(
	base_filename = "Test", 
	num_steps = 100, 
	nbatch = 2,
	seed=0, 
	ndims = 2,
	p_corrupt_attempts = 100,
	epsilon = .1,
	L = 10,
	display=1,
	mix_up = 20,
	log_conditioning_X = 2, 
	log_conditioning_P = -2,
	Ex_Type = "Gaussian_Diag",
	Ex_Structure = "Uniform",
	cov_lims = [-1,1],   
	sampler_list = ('HMC','CHMC'),
	CHMC_p_corrupt_type = 'Monte_Carlo',
	CHMC_Type = 'Quartic_Conditioned',
	beta = 1):
  

  
    ##  Seed the first point
    sp.random.seed(seed)
    file_idx = int(((time.time()*sp.random.uniform(0,1) % 10**12) % 1)*10**12)
    filename = "Data/History/%s%s,D=%s,E=%s,L=%s,Beta=%s.npz" % (base_filename, file_idx,ndims, epsilon, L, beta)
    history = OrderedDict()  
    
    ##  Save array for types of distributions including individual parameters
    target_prob_list = [Potential_Energy(ndims=ndims, 
                                         E_Type = Ex_Type, 
                                         log_conditioning=log_conditioning_X,
                                         structure = Ex_Structure,
                                         cov_lims = cov_lims,
                                         nbatch=nbatch,
                                         seed = seed)]
    
 
    print ("...Potential Energy covariance matrix created")                                                                 
    for dist_X in target_prob_list:
        dist_X_name = dist_X.description
        history[dist_X_name] = OrderedDict()
        
        if display == 1:
            print ("Covariance Matrix:\n")
            plt.matshow(dist_X.corr_matrix,cmap=plt.cm.gray)
            plt.colorbar()
            plt.show()
    
                       
        for sampler_name in sampler_list:
            print ("Generating Sampler...")
            
            if sampler_name == 'HMC':
                dist_P = Kinetic_Energy(Potential_Energy = dist_X, ndims = ndims,E_Type = "Gaussian_Diag_Inv_Scaled")    
                cw = counting_wrapper(dist_X, dist_P)
                sampler = CHMC(cw,
                               dist_X = dist_X,
                               dist_P = dist_P,
                               x_init = dist_X.x_init,#(sp.random.uniform(-1,1,size= (ndims,nbatch)).T*1./sp.diag(dist_X.x_coefficients)).T,#dist_X.x_init, 
                               epsilon=epsilon,
                               p_corrupt_attempts = p_corrupt_attempts,
                               p_corrupt_type = 'Monte_Carlo',                               
                               beta=beta,
                               seed = seed,
                               display=display,
                               num_leapfrog_steps=L, 
                               num_look_ahead_steps=1)
            
            elif sampler_name == 'CHMC':
                dist_P = Kinetic_Energy(Potential_Energy = dist_X, ndims = ndims,E_Type = CHMC_Type)    
                cw = counting_wrapper(dist_X, dist_P)
                sampler = CHMC(cw,
                               dist_X = dist_X,
                               dist_P = dist_P,
                               x_init = dist_X.x_init,#(sp.random.uniform(-1,1,size= (ndims,nbatch)).T*1./sp.diag(dist_X.x_coefficients)).T,#dist_X.x_init, 
                               epsilon=epsilon,
                               p_corrupt_attempts = p_corrupt_attempts,
                               p_corrupt_type = CHMC_p_corrupt_type,                               
                               beta=beta,
                               seed = seed,
                               display=display,
                               num_leapfrog_steps=L, 
                               num_look_ahead_steps=1,
                               mix_up = mix_up)
            else:
                raise Exception("unknown sampler %s"%(sampler_name))

            print ("...Done generating Sampler %s" %(sampler_name))
            ## Define array to hold data.
            history[dist_X_name][sampler_name] = []
            
            ## Make experiments repeatable (reseed)
            sp.random.seed(seed) 

            ## Verify form of PE and KE for user
            print("\n\nSampling from %s using %s"%(dist_X_name, sampler_name))
            if sampler.display > 0:
                print ("\nE_x Energy has the form %s" % cw.Ex_external)
                print ("E_p Energy has the form %s" % cw.Ep_external)
                print ("dEdX has the form %s" % cw.dEdX_external)
                print ("dEdP Energy has the form %s" % cw.dEdP_external)
            
            
            ## Run Monte Carlo Sampling
            timenow = time.time()
            for iN in range(num_steps):

                if math.modf(iN/num_steps*10)[0] == 0:
                    print ("%s%% ... " % int(iN/num_steps*100+10),flush=True,end= " ") 

                X,P,Ex,Ep = sampler.sample(1)

                if (sp.isnan(sp.sum(X))) or (sp.isnan(sp.sum(P))) == True:
                    for iM in range(num_steps - iN):
                          history[dist_X_name][sampler_name].append({'X':X.copy(),                                                        
								 'P':P.copy(),
	                                                         'Ex':Ex.copy(),
                                        	                 'Ep':Ep.copy(),
        	                                                 'Ex_count':cw.Ex_count,
                	                                         'Ep_count':cw.Ep_count,
                        	                                 'dEdX_count':cw.dEdX_count,
                                	                         'dEdP_count':cw.dEdP_count})

                    print ('..Invalid values ...filling with nan')
                    break
                else:
                    history[dist_X_name][sampler_name].append({'X':X.copy(), 
                                                         'P':P.copy(), 
                                                         'Ex':Ex.copy(), 
                                                         'Ep':Ep.copy(), 
                                                         'Ex_count':cw.Ex_count, 
                                                         'Ep_count':cw.Ep_count, 
                                                         'dEdX_count':cw.dEdX_count,
                                                         'dEdP_count':cw.dEdP_count})

            elapsed_time = time.time() - timenow
            print ("Time Elapsed = %s" % elapsed_time)

        save_results(history, filename)

    return filename, dist_X
