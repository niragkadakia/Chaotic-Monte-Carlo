import scipy as sp
from Includes.utils import polynomial
from Includes.utils import polynomial2
from Includes.utils import polynomial3
from Includes.utils import gen_covariance_matrix

class Potential_Energy(object):
	def __init__(self, 
							 ndims = 2, 
							 nbatch = 100,
							 E_Type = "Gaussian_Diag", 
							 structure = "Toeplitz",
							 log_conditioning = 0, 
							 cov_lims = [-1,1],
							 x_init = None,
							 seed = None): 

		"""
		Energy function, gradient, and hyperparameters for 
		potential portion of energy.
		"""
		if seed is not None:
			sp.random.seed(seed)

		self.x_coefficients = sp.zeros((ndims,ndims))
		self.log_conditioning = log_conditioning	
		self.E_Type = E_Type

		if E_Type == "Gaussian_Diag":			
			self.description = '%dD Anisotropic Gaussian, %g conditioning'%(ndims, 10**log_conditioning)
			self.polynomial_indices = sp.arange(ndims).reshape((ndims,-1))
			self.polynomial_coeff = 0.5*10**sp.linspace(-log_conditioning, 0, ndims)
			self.polynomial_powers = (sp.array([2]*ndims)).reshape((ndims,1))

			self.corr_matrix = sp.eye(ndims)
			self.cov_matrix = 10**sp.linspace(log_conditioning, 0, ndims)
			self.eigs = sp.diag(self.cov_matrix)
			self.inv_cov_matrix = sp.diag(self.polynomial_coeff*2.0)
			self.x_coefficients = self.inv_cov_matrix

		elif E_Type  == "Gaussian_Non_Diag":
			self.description = '%dD Anisotropic Non-Diagonal Gaussian, %g conditioning'%(ndims, 10**log_conditioning)

			self.corr_matrix, self.cov_matrix, self.eigs, self.inv_cov_matrix = \
				gen_covariance_matrix(log_conditioning = log_conditioning, lims = cov_lims, dims = ndims, display = 1,structure=structure,seed=seed)
			self.x_coefficients = self.inv_cov_matrix
			self.polynomial_indices,self.polynomial_coeff,self.polynomial_powers,self.coefficients_P = [],[],[],[]

			for iDx in sp.arange(ndims):
				for iDy in sp.arange(iDx):
					self.polynomial_indices.append([iDx,iDy])
					self.polynomial_coeff.append(self.inv_cov_matrix[iDx,iDy])
					self.polynomial_powers.append([1,1])

				self.polynomial_indices.append([iDx])
				self.polynomial_coeff.append(0.5*self.inv_cov_matrix[iDx,iDx])
				self.polynomial_powers.append([2])

			self.polynomial_indices = sp.asarray(self.polynomial_indices)
			self.polynomial_coeff = sp.asarray(self.polynomial_coeff)
			self.x_var_powers = sp.asarray(self.polynomial_powers)


		else:
			raise Exception ("Unknown potential energy form")

		if x_init is None:
			self.x_init = (1./sp.sqrt(sp.diagonal(self.x_coefficients)).reshape((-1,1))) * sp.random.randn(ndims,nbatch)
		else:
			self.x_init = x_init

		print ("Generating potential energy polynomial...")
		self.Ex = polynomial3(ndims, self.polynomial_indices, self.polynomial_coeff, self.polynomial_powers)
		print ("...Done")



class Kinetic_Energy(object):
	def __init__(self, 
							 Potential_Energy,
							 ndims = 2, 
							 E_Type = "Gaussian_Diag_Scaled", 
							 log_conditioning = 0):

		"""
		Energy function, gradient, and hyperparameters for 
		kinetic portion of energy.
		"""

		self.Potential_Energy = Potential_Energy
		self.p_coefficients = sp.zeros((ndims,ndims))
		self.E_Type = E_Type

		if self.E_Type == "Gaussian_Diag_Inv_Scaled":			
			self.polynomial_indices = sp.arange(ndims).reshape((ndims,-1))
			self.polynomial_coeff = 0.5/sp.diag(self.Potential_Energy.x_coefficients)
			self.polynomial_powers = (sp.array([2]*ndims)).reshape((ndims,1))
			self.p_coefficients = sp.diag(self.polynomial_coeff*2)

			self.description = '%dD Anisotropic Gaussian, %g conditioning'%(ndims, int(10**(-max(self.polynomial_coeff*2.0))))

		elif self.E_Type == "Gaussian_Diag_Custom_Scaled":
			self.polynomial_indices = sp.arange(ndims).reshape((ndims,-1))
			self.polynomial_coeff = 0.5*10**sp.linspace(-log_condioning,0,ndims)
			self.polynomial_powers = (sp.array([2]*ndims)).reshape((ndims,1))
			self.p_coefficients = sp.diag(self.polynomial_coeff*2)

			self.description = '%dD Anisotropic Gaussian, %g conditioning'%(ndims, 10**log_conditioning)

			
		
		elif self.E_Type == "Quartic_Conditioned":
			self.description = '%dD Quartic, %g conditioning'%(ndims,10**log_conditioning)

			A,B,C = [],[],[]
			self.p_coefficients = sp.diag(1/sp.diag(self.Potential_Energy.x_coefficients))

			for idx in sp.arange(ndims):
				mystr = "A.append([%s])" % idx
				exec (mystr)
				B.append(0.5*self.p_coefficients[idx,idx])
				C.append([2])
			for idx in sp.arange(ndims):
				##Nearest neighbor coupling
				#for idy in sp.arange(ndims):
					#if idy == ((idx + 1) % ndims):
                                        #        mystr = "A.append([%s,%s])" % (idx,idy)
                                        #        exec(mystr)
                                        #        B.append(.5*self.p_coefficients[idx,idx]*self.p_coefficients[idy,idy])
                                        #        C.append([2,2])
				
				##All to all coupling
				for idy in sp.arange(idx):
					if (idx != idy):
						mystr = "A.append([%s,%s])" % (idx,idy)
						exec(mystr)
						B.append(.5*self.p_coefficients[idx,idx]*self.p_coefficients[idy,idy])
						C.append([2,2])			

			self.polynomial_indices = sp.asarray(A)
			self.polynomial_coeff = sp.asarray(B)
			self.polynomial_powers = sp.asarray(C)

		elif self.E_Type == "Quartic_Conditioned_Separable":
			self.description = '%dD Quartic, %g conditioning, Separable'%(ndims,10**log_conditioning)

			A,B,C = [],[],[]
			self.p_coefficients = sp.diag(1/sp.diag(self.Potential_Energy.x_coefficients))

			for idx in sp.arange(ndims):
				mystr = "A.append([%s])" % idx
				exec (mystr)
				B.append(0.5*self.p_coefficients[idx,idx])
				C.append([2])
			for idx in sp.arange(ndims):
				for idy in sp.arange(1,ndims,2):
					if idy == ((idx + 1) % ndims):
						mystr = "A.append([%s,%s])" % (idx,idy)
						exec(mystr)
						B.append(.5*self.p_coefficients[idx,idx]*self.p_coefficients[idy,idy])
						C.append([2,2])			

			self.polynomial_indices = sp.asarray(A)
			self.polynomial_coeff = sp.asarray(B)
			self.polynomial_powers = sp.asarray(C)


		else:
			raise Exception ("Unknown kinetic energy form")

		## self.Ep holds all symbolic information about the kinetic energy F and its gradient dF
		print ("Generating kinetic energy polynomial...")
		self.Ep = polynomial3(ndims, self.polynomial_indices, self.polynomial_coeff, self.polynomial_powers)
		print ("...Done")

