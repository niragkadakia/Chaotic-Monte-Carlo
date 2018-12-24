import scipy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

colors = ['red','blue','green']


def calc_correlation(hist_single):
	T = len(hist_single)
	N = hist_single[0]['X'].shape[0]
	nbatch = hist_single[0]['X'].shape[1]
	X = sp.zeros((N,nbatch,T))

	for tt in range(T):
		  X[:,:,tt] = hist_single[tt]['X']

	c = sp.zeros((T,))
	c[0] = sp.mean(X[:,:,:]**2)

	for t_gap in range(1, T):
		if (t_gap % (int(T/10)) == 0):
			print ("%s correlations calculated..." %(t_gap))
		c[t_gap] = sp.mean(X[:,:,:-t_gap]*X[:,:,t_gap:])
	return c/c[0]


def plot_correlation(history):

  for dist_name in list(history.keys()):
      fig = plt.figure()
      fig.set_size_inches(14,6)
      corr_cum = [] 
      for samp_name in list(history[dist_name].keys()):
          hist_single = history[dist_name][samp_name]
          nsteps = len(hist_single)
          nbatch = hist_single[-1]['X'].shape[1]
          nfunc = hist_single[-1]['dEdX_count']
          corr = calc_correlation(hist_single)
          t_diff = sp.linspace(0, nfunc-1, nsteps+1)
          corr_cum.append(t_diff[1:])
          corr_cum.append(corr)
          plt.plot(t_diff[1:], corr, label=samp_name)
      corr_cum  = sp.array(corr_cum)
      corr_to_file = sp.zeros((len(corr_cum), nsteps))
      for i in sp.arange(len(corr_cum)):
            corr_to_file[i,:] = corr_cum[i]
      plt.legend()
      plt.xlabel('Gradient Evaluations')
      plt.ylabel('Autocorrelation')
      plt.title(dist_name)
      plt.grid()
      plt.savefig('Figures/Figure_%s.jpg' % dist_name)
  plt.show()
  return corr_to_file
        

def calc_covariance_errors(history, dist_X):
    
    print ('Calculating covariance errors...')
    for dist_name in list(history.keys()):
        nTypes = len(history[dist_name].keys())
        hist_single = history[dist_name][list(history[dist_name].keys())[0]]
        nsteps = len(hist_single)
        samp_names = []
        errors = sp.zeros((nsteps,nTypes,2))

        counter = 0
        for samp_name in list(history[dist_name].keys()): 
            samp_names.append(samp_name)
            hist_single = history[dist_name][samp_name]
            nsteps = len(hist_single)
            nbatch = hist_single[-1]['X'].shape[1]
            N = hist_single[0]['X'].shape[0]
            errors_tmp = sp.zeros((nsteps,nTypes))

            X = sp.zeros((N,nbatch,nsteps))
            P = sp.zeros((N,nbatch,nsteps))
            for tt in range(nsteps):
                X[:,:,tt] = hist_single[tt]['X']
                P[:,:,tt] = hist_single[tt]['P']


            inv_var_diags = 10.**sp.linspace(-dist_X.log_conditioning, 0, N)
            corr_matrix_calc = sp.zeros((N,N,nsteps))
            cov_matrix_calc = sp.zeros((N,N,nsteps))

            for iN in sp.arange(1,nsteps):

              if (iN % (nsteps/10) == 0):
                print ("%s: %s errors calculated..." %(samp_name, iN))
              
              cov_matrix_calc[:,:,iN] = sp.cov(X[:,:,:iN].reshape(N,nbatch*iN),rowvar=1)
              corr_matrix_calc[:,:,iN] = sp.dot(sp.dot(sp.diag(inv_var_diags**.5),cov_matrix_calc[:,:,iN]),sp.diag(inv_var_diags**.5))
              errors_tmp[iN,0] = sp.sum((sp.diag(sp.diag(corr_matrix_calc[:,:,iN]))-sp.diag(sp.diag(dist_X.corr_matrix)))**2.0)/N
              errors_tmp[iN,1] = sp.sum((corr_matrix_calc[:,:,iN] - sp.diag(sp.diag(corr_matrix_calc[:,:,iN]))-dist_X.corr_matrix +sp.diag(sp.diag(dist_X.corr_matrix)))**2.0)/(N*(N-1))

            print (corr_matrix_calc[:5,:5,-1])
            print (dist_X.corr_matrix[:5,:5])
               
            errors[:,counter,0] = errors_tmp[:,0]            
            errors[:,counter,1] = errors_tmp[:,1]            

            counter += 1

    return errors, samp_names
	
 
def plot_covariance(history, dist_X):
   
    for dist_name in list(history.keys()):
        nTypes = len(history[dist_name].keys())
        errors = sp.zeros((2,nTypes))
        fig = plt.figure()
        fig.set_size_inches(6*nTypes,5)       
        plt.subplot(1,nTypes+1,1)
        plt.imshow(dist_X.corr_matrix,cmap=plt.cm.gray,interpolation='none')

        counter = 0
        for samp_name in list(history[dist_name].keys()):
            counter += 1
            hist_single = history[dist_name][samp_name]
            nsteps = len(hist_single)
            nbatch = hist_single[-1]['X'].shape[1]
            N = hist_single[0]['X'].shape[0]

            X = sp.zeros((N,nbatch,nsteps))
            P = sp.zeros((N,nbatch,nsteps))
            for tt in range(nsteps):
                X[:,:,tt] = hist_single[tt]['X']
                P[:,:,tt] = hist_single[tt]['P']
                
            ax = plt.subplot(1,nTypes+1,counter+1)
            inv_var_diags = sp.diag(10.**sp.linspace(-dist_X.log_conditioning, 0, N))**.5
            corr_matrix_calc = sp.dot(sp.dot(inv_var_diags**.5,sp.cov(X.reshape(N,nbatch*nsteps),rowvar = 1)),inv_var_diags**.5)
            plt.imshow(corr_matrix_calc,cmap=plt.cm.gray,interpolation='none')
           
            print (corr_matrix_calc)
        plt.show()
