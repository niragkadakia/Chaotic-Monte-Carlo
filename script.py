import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import pandas as pd



def phase_plots():
	file_names = ['Allstates_NonChaotic_Stimulus','Allstates_Chaotic_Stimulus']
	vars_to_plot = [1,2]
	data_to_plot = -1

	for filename in file_names:
		data = sp.loadtxt('%s.dat' % filename)
		fig = plt.figure()
		fig.set_size_inches(6,5)
		plt.plot(data[:data_to_plot,vars_to_plot[0]], data[:data_to_plot,vars_to_plot[1]],color = plt.cm.Blues(.95))
		plt.tick_params(axis='x', labelsize=13)
		plt.tick_params(axis='y', labelsize=13)
		plt.ylabel(r'$n(t)$', fontsize = 20)
		plt.xlabel(r'$V_{soma}(t)$',fontsize = 20)
		plt.xlim(-100,40)
		plt.tight_layout()
		plt.savefig('%s_%s_%s.svg' % (filename,vars_to_plot[0],vars_to_plot[1]),bbox_inches='tight')
		plt.savefig('%s_%s_%s.png' % (filename,vars_to_plot[0],vars_to_plot[1]),bbox_inches='tight')
		plt.show()





def current_traces():
	file_names = ['Current_Dendrite_Chaotic_Stimulus']#,'Current_Dendrite_Chaotic_Stimulus']
	data_to_plot = 5000

	dt = 0.2

	for filename in file_names:
		data = sp.loadtxt('%s.dat' % filename)
		fig = plt.figure()
		fig.set_size_inches(7,3.5)
		ax = plt.subplot(111)
		if data_to_plot == -1:
			plt.plot(sp.linspace(0,dt*(len(data)-1),(len(data)-1)), data[:data_to_plot],color='r',linewidth=2	)
		else:
			plt.plot(sp.linspace(0,data_to_plot*dt,data_to_plot), data[:data_to_plot],color='r',linewidth=2)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		plt.tick_params(
			axis='both',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom='off',      # ticks along the bottom edge are off
			top='off',         # ticks along the top edge are off
			left='off',      # ticks along the bottom edge are off
			right='off',         # ticks along the top edge are off
			labelbottom='off', # labels along the bottom edge are off
			labelleft='off') # labels along the bottom edge are off
#		plt.tick_params(axis='x', labelsize=13)
#		plt.tick_params(axis='y', labelsize=13)
#		ax.xaxis.set_ticks_position('bottom')
#		ax.yaxis.set_ticks_position('left')
		plt.xlabel(r'$t$', fontsize = 26)
		plt.ylabel(r'$I_{injected}(t)$',fontsize = 26)
		plt.tight_layout()
		plt.savefig('%s.svg' % (filename),bbox_inches='tight')
		plt.savefig('%s.png' % (filename),bbox_inches='tight')
		plt.show()





def est_and_pred_current_traces():
	input_dir = 'Chaotic_Predictions/Assimilation_Data'
	file_names_pred = ['%s/Chaotic_Current/current' % input_dir, '%s/Step_Current/current' % input_dir]
	input_dir = 'Chaotic_Predictions/Prediction_Data'
	file_names_current = '%s/current_True_pred' % input_dir
	pred_current = sp.loadtxt('%s.dat' % file_names_current)

	dt = 0.02
	nT = 6000

	time_pred = sp.linspace(nT*dt, nT*dt + len(pred_current)*dt, len(pred_current))

	for filename in file_names_pred:
		data_cur = sp.loadtxt('%s.dat' % filename)

		fig = plt.figure()
		fig.set_size_inches(7,3.5)
		ax = plt.subplot(111)
		plt.plot(sp.linspace(0,nT*dt,nT), data_cur[:nT],color='r',linewidth=2)
		plt.plot(time_pred, pred_current,color='r',linewidth=2)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		plt.tick_params(
			axis='both',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom='off',      # ticks along the bottom edge are off
			top='off',         # ticks along the top edge are off
			left='off',      # ticks along the bottom edge are off
			right='off',         # ticks along the top edge are off
			labelbottom='off', # labels along the bottom edge are off
			labelleft='off') # labels along the bottom edge are off
		plt.axvline(nT*dt, color = 'y',linewidth=3,linestyle = '--')

		plt.xlabel(r'$t$', fontsize = 26)
		plt.ylabel(r'$I_{injected}(t)$',fontsize = 26)
		plt.tight_layout()
		plt.savefig('%s.svg' % (filename),bbox_inches='tight')
		plt.savefig('%s.png' % (filename),bbox_inches='tight')
		plt.show()




def est_and_pred_voltage_traces():
	input_dir = 'Chaotic_Predictions/Assimilation_Data'
	file_names_est = ['%s/Chaotic_Current/Estimation29' % input_dir, '%s/Step_Current/Estimation29' % input_dir]
	file_names_true = ['%s/Chaotic_Current/allstates' % input_dir, '%s/Step_Current/allstates' % input_dir]
	input_dir = 'Chaotic_Predictions/Prediction_Data'
	file_names_pred = '%s/voltage_prediction_data_to_file' % input_dir # Format of this file: Time | Real | Chaotic Pred | Step Pred

	dt = 0.02
	nT = 6000
	
	data_pred = sp.loadtxt('%s.dat' % file_names_pred)

	for iF in sp.arange(len(file_names_est)):
		fig = plt.figure()
		fig.set_size_inches(7,3.5)
		ax = plt.subplot(111)

		data_est = sp.loadtxt('%s.dat' % file_names_est[iF])
		plt.plot(sp.linspace(0,nT*dt,nT), data_est[:nT,1],linewidth=3,color = plt.cm.Blues(.95)) #Est
		plt.plot(nT*dt + data_pred[:,0], data_pred[:,2+iF],linewidth=3,color = plt.cm.Blues(.95)) #Pred

		data = sp.loadtxt('%s.dat' % file_names_true[iF])
		plt.plot(sp.linspace(0,nT*dt,nT), data[:nT,1], color = 'r', marker = 'o', linewidth = .1, markevery = 10,markersize = 3) #True Est
		plt.plot(nT*dt + data_pred[:,0], data_pred[:,1], color = 'r', marker = 'o', linewidth = .1, markevery = 10,markersize = 3) #True Pred

		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		plt.tick_params(
			axis='both',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom='off',      # ticks along the bottom edge are off
			top='off',         # ticks along the top edge are off
			left='off',      # ticks along the bottom edge are off
			right='off',         # ticks along the top edge are off
			labelbottom='off', # labels along the bottom edge are off
			labelleft='off') # labels along the bottom edge are off
		plt.xlabel(r'$t$', fontsize = 26)
		plt.ylabel(r'$V(t)$',fontsize = 26)
		plt.axvline(nT*dt, color = 'y',linewidth=3,linestyle = '--')
		plt.tight_layout()
		plt.savefig('%s_with_prediction.svg' % (file_names_est[iF]),bbox_inches='tight')
		plt.savefig('%s_with_prediction.png' % (file_names_est[iF]),bbox_inches='tight')
		plt.show()




def anneal_plots():
	input_dir = 'Anneal'
	data = sp.loadtxt('%s/Action_Levels.dat' % input_dir)
	colors = ['b','r','g','c', 'm', 'y']
	set_to_plot = [59,12,23]
	log_Rf0 = -6

	fig = plt.figure()
	fig.set_size_inches(7,7)
	plt.tick_params(axis='x', labelsize=13)
	plt.tick_params(axis='y', labelsize=13)
	plt.ylabel(r'$\log[A({\bf X})]$', fontsize = 20)
	plt.xlabel(r'$\log(R_f)$',fontsize = 20)
	plt.xlim(-7,44)

	for iN in set_to_plot:
		plt.plot(sp.arange(len(data[:,0])) + log_Rf0 , sp.log(data[:,iN])/sp.log(10),marker = 'o')

	plt.savefig('%s/Action_Levels.svg' %input_dir,bbox_inches='tight')
	plt.savefig('%s/Action_Levels.png' %input_dir,bbox_inches ='tight')
	plt.show()


def anneal_estimates():
	input_dir = 'Anneal/'
	beta = 32
	colors = ['b','r','g','c', 'm', 'y']
	colors = ['g','r','b']
	w1,w2 = 0.5,1.0
	var_to_plot = 0
	file_names = ['%sEstimation%s_3rdmin' % (input_dir, beta), '%sEstimation%s_2ndmin' % (input_dir, beta), '%sEstimation%s_1stmin' % (input_dir, beta)]
	data_true = sp.loadtxt('%sallstates.dat' % input_dir)
	data_meas = sp.loadtxt('%stwin_data.dat' % input_dir)

	num_files = len(file_names)
	fig = plt.figure()
	fig.set_size_inches(2.4,6)
	for iF in sp.arange(len(file_names)):
		ax = plt.subplot(num_files,1,iF+1)
		data_est = sp.loadtxt('%s.dat' % file_names[iF])
		N = len(data_est[:,0])
		dt = data_est[1,0] - data_est[0,0]

		plt.plot(data_est[N*w1:N*w2,0], data_est[N*w1:N*w2, var_to_plot+1],linewidth=3,color = colors[iF])
		plt.plot(data_est[N*w1:N*w2,0], data_true[N*w1:N*w2,var_to_plot], color = 'k',linewidth = 2)

		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		plt.tick_params(
			axis='both',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom='off',      # ticks along the bottom edge are off
			top='off',         # ticks along the top edge are off
			left='off',      # ticks along the bottom edge are off
			right='off',         # ticks along the top edge are off
			labelbottom='off', # labels along the bottom edge are off
			labelleft='off') # labels along the bottom edge are off
		plt.xlabel(r'$t$', fontsize = 26)
		plt.ylabel(r'$V(t)$',fontsize = 26)
		plt.xlim(N*w1*dt,(N-1)*w2*dt)
	plt.tight_layout()
	plt.savefig('%sEstimation%s_var=%s.svg' % (input_dir, beta, var_to_plot),bbox_inches='tight')
	plt.savefig('%sEstimation%s_var=%s.png' % (input_dir, beta, var_to_plot),bbox_inches='tight')
	plt.show()


def plot_polynomial():

	beta = 0

	input_dir = 'Anneal'
	colors = ['r','g','b']
	x = sp.arange(-10,10,0.01)
#	y = x*(x-5)*(x-3)*(x+2)*(x-0.5)*(x+4)
#	y = x*(x-5)*(x-3)*(x+.15)
	y = x**2.0

	if min(y) < 0:
		y = y - min(y)

	fig = plt.figure()
	fig.set_size_inches(4,5)
	plt.plot(x,y,color = 'k',linewidth=2)

	min_vals = []
#	partitions = sp.array([[500,700],[900,1200],[1400,1500]])
#	partitions = sp.array([[900,1200],[900,1200],[1400,1500]])
	partitions = sp.array([[900,1200],[900,1200],[900,1200]])
	for iP in partitions:
		min_vals.append(y[iP[0]:iP[1]].argsort()[0]+iP[0])
	min_vals[0]+= 40
	min_vals[1]-= 40
	min_vals[2]-= 80
	print (min_vals)
	for iN in sp.arange(len(min_vals)):
		plt.scatter(x[min_vals[iN]], y[min_vals[iN]],color=colors[iN],s=250)

	plt.ylim(-5,50)
	plt.xlim(-8,8)
	plt.xlabel(r'${\bf X}$', fontsize = 26)
	plt.ylabel(r'$A({\bf X})$',fontsize = 26)
	plt.tick_params(
		axis='both',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom='off',      # ticks along the bottom edge are off
		top='off',         # ticks along the top edge are off
		left='off',      # ticks along the bottom edge are off
		right='off',         # ticks along the top edge are off
		labelbottom='off', # labels along the bottom edge are off
		labelleft='off') # labels along the bottom edge are off

	plt.savefig('%s/Action_Surface_%s.svg' % (input_dir,beta),bbox_inches='tight')
	plt.savefig('%s/Action_Surface_%s.png' % (input_dir,beta),bbox_inches='tight')
	plt.show()


def plot_network_traces():

	input_dir = "Network/5x5"
	num_nodes = 5
	var = 0
	length = 2000

	pred_data = (pd.read_csv('%s/prediction_states.dat' % input_dir, sep = '\t')).values


	for iN in sp.arange(0,num_nodes*4,4):
		fig = plt.figure()
		fig.set_size_inches(7,3.5)
		plt.plot(pred_data[:length,0], pred_data[:length,var + 1+4*num_nodes+iN], color = plt.cm.Blues(.95),linewidth = 3)
		plt.plot(pred_data[:length,0], pred_data[:length,var + 1+iN], color = 'r',linewidth = .3, marker = 'o', markersize=3)
#		plt.tick_params(axis='x', labelsize=14)
#		plt.tick_params(axis='y', labelsize=14)
		plt.ylabel(r'$V_%s(t)$' % int(iN/4), fontsize = 26)
		plt.xlabel('$t$',fontsize = 26)
		plt.tick_params(
			axis='both',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom='off',      # ticks along the bottom edge are off
			top='off',         # ticks along the top edge are off
			left='off',      # ticks along the bottom edge are off
			right='off',         # ticks along the top edge are off
			labelbottom='off', # labels along the bottom edge are off
			labelleft='off') # labels along the bottom edge are off
		plt.tight_layout()
	
		plt.savefig('%s/predictions_%s.png' % (input_dir, iN))
		plt.savefig('%s/predictions_%s.svg' % (input_dir, iN))
		plt.show()


def plot_connectivity():

	input_dir = "Network/5x5"
	myfile = "Connectivity_True"


	connectivity = sp.loadtxt('%s/%s.dat' % (input_dir, myfile))

	plt.matshow(connectivity, interpolation='nearest', cmap=plt.cm.gray)
	plt.tick_params(
		axis='both',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom='off',      # ticks along the bottom edge are off
		top='off',         # ticks along the top edge are off
		left='off',      # ticks along the bottom edge are off
		right='off',         # ticks along the top edge are off
		labelleft='off', # labels along the bottom edge are off
		labeltop='off') # labels along the bottom edge are off
	plt.savefig('%s/%s.svg' % (input_dir,myfile))
	plt.savefig('%s/%s.png' % (input_dir,myfile))
	plt.show()

def plot_connectivity_plus_values():

	num_nodes = 6
	input_dir = "Network/6x6-0"
	myfile = "Connectivity_True"


	connectivity = sp.loadtxt('%s/%s.dat' % (input_dir, myfile))

	fig, ax = plt.subplots()
	fig.set_size_inches(6,6)

	plt.imshow(connectivity, interpolation='nearest', cmap=plt.cm.Blues,\
			vmin = -sp.amax(connectivity), vmax = sp.amax(connectivity*1.5), extent=[0,num_nodes,0,num_nodes])

	for iN in sp.arange(num_nodes):
		for jN in sp.arange(num_nodes):
			ax.text(iN+.5, num_nodes - jN-.5, round(connectivity[jN,iN],1), fontweight="bold",va='center', ha='center',fontsize = 18,color='k')

	ax.set_xlim(0,num_nodes)
	ax.set_ylim(0,num_nodes)
	plt.tick_params(
		labelbottom='off', # labels along the bottom edge are off
		labelleft='off') # labels along the bottom edge are off

	plt.savefig('%s/%s_with_values.svg' % (input_dir,myfile),bbox_inches='tight')
	plt.savefig('%s/%s_with_values.png' % (input_dir,myfile),bbox_inches='tight')
	plt.show()






################################### 			PLOT		#########################

#current_traces()
#phase_plots()
#est_and_pred_current_traces()
#est_and_pred_voltage_traces()
#anneal_plots()
#anneal_estimates()
#plot_polynomial()
#plot_network_traces()
#plot_connectivity()
plot_connectivity_plus_values()



