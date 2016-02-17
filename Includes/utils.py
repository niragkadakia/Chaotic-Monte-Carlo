import scipy as sp
from scipy.linalg import inv
from scipy.linalg import toeplitz
import sympy as sy
import re

num2words = {1: 'On0e', 2: 'Tw0o', 3: 'Th0ree', 4: 'Fo0ur', 5: 'Fi0ve', \
         6: 'Si0x', 7: 'Se0ven', 8: 'Ei0ght', 9: 'Ni0ne', 10: 'Te0n', \
        11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fo1urteen', \
        15: 'Fifteen', 16: 'Si1xteen', 17: 'Se1venteen', 18: 'Ei1ghteen', \
        19: 'Ni1neteen', 20: 'Tw2enty', 21: 'Twentyone', 22: 'Twentytwo', \
	23: 'Twentythree', 24: 'Twentyfour', 25: "Twentyfive", 26: 'Twentysix', \
        27: 'Twentyseven', 28: 'Twentyeight', 29: 'Twentynine', 30: 'Th3irty', \
        31: 'Thirtyone', 32: 'Thirtytwo', 33: 'Thirtythree', 34: 'Thirtyfour', \
        35: 'Thirtyfive', 36: 'Thirtysix', 37: 'Thirtyseven', 38: 'Thirtyeight', \
        39: 'Thirtynine', 40: 'Fo4rty', 41: 'Fortyone', 42: 'Fortytwo', 43: 'Fortythree', \
        44: 'Fortyfour', 45: 'Fortyfive', 46: 'Fortysix', 47: 'Fortyseven', \
        48: 'Fortyeight', 49: 'Fortynine', \
        50: 'Fi5fty', 51: 'Fiftyone', 52: 'Fiftytwo', 53: 'Fiftythree', 54: 'Fiftyfour', \
        55: 'Fiftyfive', 56: 'Fiftysix', 57: 'Fiftyseven', 58: 'Fiftyeight', 59: 'Fiftynine', \
        60: 'Si6xty', 61: 'Sixtyone', 62: 'Sixtytwo', 63: 'Sixtythree', 64: 'Sixtyfour', \
        65: 'Sixtyfive', 66: 'Sixtysix', 67: 'Sixtyseven', 68: 'Sixtyeight', 69: 'Sixtynine', \
        70: 'Se7venty', 71: 'Seventyone', 72: 'Seventytwo', 73: 'Seventythree', 74: 'Seventyfour', \
        75: 'Seventyfive', 76: 'Seventysix', 77: 'Seventyseven', 78: 'Seventyeight', 79: 'Seventynine', \
        80: 'Eig8hty', 81: 'Eightyone', 82: 'Eightytwo', 83: 'Eightythree', 84: 'Eightyfour', \
        85: 'Eightyfive', 86: 'Eightysix', 87: 'Eightyseven', 88: 'Eightyeight', 89: 'Eightynine', \
        90: 'Ni9nety', 91: 'Ninetyone', 92: 'Ninetytwo', 93: 'Ninetythree', 94: 'Ninetyfour', \
        95: 'Ninetyfive', 96: 'Ninetysix', 97: 'Ninetyseven', 98: 'Ninetyeight', 99: 'Ninetynine', \
        100: 'Onehundred', 0: 'Zero'}



def diag_normal_pdf(x,variance,batch):
	""" 
	Method to sample from diagonal multivariate normal distribution
	"""
	if sp.array_equal(variance, sp.ones(sp.shape(x)[0])):
		pdf_tmp = sp.exp(-(x**2.0)/2.).T/(2.*sp.pi)**.5
	else:
		var_batch = (sp.ones((sp.shape(x)[0],batch)).T*variance).T
		pdf_tmp = sp.exp(-(x**2.0)/2./var_batch).T/(2.*sp.pi*variance)**.5
	return (sp.prod(pdf_tmp,axis=1))



def gen_covariance_matrix(log_conditioning = 2, lims=[-.5,.5],dims=2,display = 0,conditioning =0, structure = "Toeplitz",seed=None):
	"""
	Method to generate covariance matrices from random correlation matrices
	"""

	print ("Finding Feasible Covariance Matrix...")

	variance = 10**sp.linspace(log_conditioning, 0, dims)
	pos_semi_def = False
	counter = 0		

	if seed is not None:
		sp.random.seed(seed)

	while (pos_semi_def == False):
		if structure == "Toeplitz_Exponential":
			noise_factor = 3
			rho = sp.random.uniform(-1,1)
			toeplitz_vals = rho**sp.arange(dims)
			corr_matrix = toeplitz(toeplitz_vals)*sp.random.normal(1,abs(rho/noise_factor),size=(dims,dims))
			print ("Rho = %s" %rho)

		elif structure == "Toeplitz_Linear":
			noise_factor = 3
			rho = sp.random.uniform(-1,1)
			toeplitz_vals = rho/sp.arange(1,dims+1)
			corr_matrix = toeplitz(toeplitz_vals)*sp.random.normal(1,abs(rho/noise_factor),size=(dims,dims))
			print ("Rho = %s" %rho)


		elif structure == "Uniform":
			corr_matrix = sp.random.uniform(lims[0],lims[1],size=(dims,dims))


		else: 
			raise Exception ("Unknown Correlation Matrix Structure")
	
		sp.fill_diagonal(corr_matrix,sp.ones(dims))
		corr_matrix = (corr_matrix+corr_matrix.T)/2.0
		cov_matrix = sp.dot(sp.dot(sp.diag(variance**.5),corr_matrix),sp.diag(variance**.5))
		eigs  = sp.real(sp.linalg.eig(cov_matrix)[0])
		pos_semi_def = (sp.sum(eigs < 0) == 0)

		counter += 1
		if (display == 1) and (counter % 50 == 0):
			print ("%s attempts..." %(counter),flush=True,end = " ")


	inv_cov_matrix = sp.linalg.inv(cov_matrix)    

	print ("...done")

	return (corr_matrix, cov_matrix, eigs, inv_cov_matrix)


class polynomial:
	def __init__(self, ndims, var_indices, coefficients, powers):
		"""
		Class to write polynomials and their derivatives
		  of various order and shape for the distributions

		var_indices is an n-array of arrays, each of which indicates
		  which x[m] factors will exist in the nth term
		coefficients is an n-array which indicates what are the 
			coefficients of the n-term
		powers is an n-array of arrays, each of which indicates
			the powers of x[m] in the nth term.

		This uses sympy to analytically evaluate the derivatives 
		-- which can be very slow.
		"""


		self.F = ""
		self.dF = []

		## Write string for the polynomial expression
		if len(var_indices) == len(coefficients) == len(powers):

			for iTerm in sp.arange(len(coefficients)):
				if len(var_indices[iTerm]) == len(powers[iTerm]):
					self.F += "%s*" % coefficients[iTerm]
					for iFactor in sp.arange(len(var_indices[iTerm])):
						self.F += "X[%s]**%s*" % (var_indices[iTerm][iFactor], powers[iTerm][iFactor])
				else:
					raise Exception ("variable index and and powers arrays \
												   in one of the factors of the polynomial \
													 do not match")
				self.F = self.F[:-1] + " + "
			self.F = self.F[:-2]
		else:
			raise Exception ("Wrong number of terms of either coefficients, \
												powers or variables in polynomial arrays")

		## Write string for the derivatives using sympy
		print ("    Writing symbolic strings...")
		tmp_str = self.F
		for iN in sp.arange(ndims):
			tmp_str = re.sub('X\[%s\]' % iN, 'X%s' % re.sub("-","",num2words[iN]) , tmp_str)
			exec("X%s = sy.Symbol('X%s')" % (re.sub("-","",num2words[iN]), re.sub("-","",num2words[iN])))
		print ("    ...done")

		print (tmp_str)

		print ("    Generating derivative strings...")
		self.F_Symbolic = tmp_str
		for iN in sp.arange(ndims):
				exec(compile("self.tmp_str = str(sy.diff(self.F_Symbolic, X%s))" % re.sub("-","",num2words[iN]),'string','exec'))				
				print ("    ...%s" % iN) 
				for iM in sp.arange(ndims):
						self.tmp_str = re.sub('X%s' % re.sub("-", "", num2words[iM]), 'X[%s]' % iM, self.tmp_str)
				self.dF.append(self.tmp_str)
		print ("    ...done")



class polynomial2:
	def __init__(self, ndims, var_indices, coefficients, powers):
		"""
		Class to write polynomials and their derivatives 
			of various order and shape for the distributions

		var_indices is an n-array of arrays, each of which indicates
		  which x[m] factors will exist in the nth term
		coefficients is an n-array which indicates what are the 
			coefficients of the n-term
		powers is an n-array of arrays, each of which indicates
			the powers of x[m] in the nth term.
		The derivatives in this module are put in by hand
			an then simplified using sympy 
		"""


		self.F = ""
		self.dF_symbolic = ['']*ndims
		self.dF = ['']*ndims

		## Write string for the polynomial expression
		if len(var_indices) == len(coefficients) == len(powers):

			for iTerm in sp.arange(len(coefficients)):
				tmp_F = ""
				if len(var_indices[iTerm]) == len(powers[iTerm]):

					## F
					tmp_F += "%s*" % coefficients[iTerm] # This will not work for mixed log/polynomial terms
					for iFactor in sp.arange(len(var_indices[iTerm])):
						tmp_F += "X[%s]**%s*" % (var_indices[iTerm][iFactor], powers[iTerm][iFactor])
					tmp_F = tmp_F[:-1] # Remove trailing *
					self.F += tmp_F

					## dF
					for iFactor in sp.arange(len(var_indices[iTerm])):						
						self.dF_symbolic[var_indices[iTerm][iFactor]] += "+%s/X[%s]*%s" %(tmp_F,var_indices[iTerm][iFactor],powers[iTerm][iFactor])
				else:
					raise Exception ("variable index and and powers arrays \
												   in one of the factors of the polynomial \
													do not match")
				self.F = self.F + " + "
			self.F = self.F[:-2]
		else:
			raise Exception ("Wrong number of terms of either coefficients, \
												powers or variables in polynomial arrays")

		## Simplify string for the derivatives using sympy
		print ("    Sympifying dF...")

		for iM in sp.arange(ndims):
			for iN in sp.arange(ndims):
				self.dF_symbolic[iM] = re.sub('X\[%s\]' % iN, 'X%s' % re.sub("-","",num2words[iN]) , self.dF_symbolic[iM])
				exec("X%s = sy.Symbol('X%s')" % (re.sub("-","",num2words[iN]), re.sub("-","",num2words[iN])))
			print ("%s..." %iM,flush=True,end = " ")
			self.dF_symbolic[iM] = sy.sympify(self.dF_symbolic[iM])
			self.dF[iM] = "%s" % self.dF_symbolic[iM]

		for iM in sp.arange(ndims):
				for iN in sp.arange(ndims):
					self.dF[iM] = re.sub('X%s' % re.sub("-", "", num2words[iN]), 'X[%s]' % iN, self.dF[iM])

		print ("    ...done")



class polynomial3:
        def __init__(self, ndims, var_indices, coefficients, powers):
                """
                Class to write polynomials and their derivatives
                        of various order and shape for the distributions

                var_indices is an n-array of arrays, each of which indicates
                  which x[m] factors will exist in the nth term
                coefficients is an n-array which indicates what are the
                        coefficients of the n-term
                powers is an n-array of arrays, each of which indicates
                        the powers of x[m] in the nth term.
                
		The derivatives in this module are put in by hand
                        an then simplified using sympy

		This module also handles log terms using the power of 0
                """


                self.F = ""
                self.dF_symbolic = ['']*ndims
                self.dF = ['']*ndims

                ## Write string for the polynomial expression
                if len(var_indices) == len(coefficients) == len(powers):

                        for iTerm in sp.arange(len(coefficients)):
                                tmp_F = ""
                                if len(var_indices[iTerm]) == len(powers[iTerm]):

                                        ## F
                                        if sp.product(powers[iTerm][:]) != 0:
                                                tmp_F += "%s*" % coefficients[iTerm] 
                                        else:
                                                tmp_F += "2.0*" ## This will not work for mixed log/polynomial
                                        for iFactor in sp.arange(len(var_indices[iTerm])):
                                                if powers[iTerm][iFactor] == 0:
                                                      tmp_F += "-log(abs(%s*X[%s]))*" % (coefficients[iTerm],var_indices[iTerm][iFactor])
                                                else:
                                                      tmp_F += "X[%s]**%s*" % (var_indices[iTerm][iFactor], powers[iTerm][iFactor])
                                        tmp_F = tmp_F[:-1] # Remove trailing *
                                        self.F += tmp_F

                                        ## dF
                                        for iFactor in sp.arange(len(var_indices[iTerm])):
                                                if powers[iTerm][iFactor] == 0:
                                                      self.dF_symbolic[var_indices[iTerm][iFactor]] += "+%s/log(abs(%s*X[%s]))/X[%s]" %(tmp_F,coefficients[iTerm],var_indices[iTerm][iFactor],var_indices[iTerm][iFactor])
                                                else:
                                                      self.dF_symbolic[var_indices[iTerm][iFactor]] += "+%s/X[%s]*%s" %(tmp_F,var_indices[iTerm][iFactor],powers[iTerm][iFactor])
                                else:
                                        raise Exception ("variable index and and powers arrays \
                                                                                                   in one of the factors of the polynomial \
                                                                                                        do not match")
                                self.F = self.F + " + "
                        self.F = self.F[:-2]
                else:
                        raise Exception ("Wrong number of terms of either coefficients, \
                                                                                                powers or variables in polynomial arrays")

                ## Simplify string for the derivatives using sympy
                print ("    Sympifying dF...")


                for iM in sp.arange(ndims):
                        for iN in sp.arange(ndims):
                                self.dF_symbolic[iM] = re.sub('X\[%s\]' % iN, 'X%s' % re.sub("-","",num2words[iN]) , self.dF_symbolic[iM])
                                exec("X%s = sy.Symbol('X%s')" % (re.sub("-","",num2words[iN]), re.sub("-","",num2words[iN])))
                        print ("%s..." %iM,flush=True,end = " ")
                        self.dF_symbolic[iM] = sy.sympify(self.dF_symbolic[iM])
                        self.dF[iM] = "%s" % self.dF_symbolic[iM]

                for iM in sp.arange(ndims):
                                for iN in sp.arange(ndims):
                                        self.dF[iM] = re.sub('X%s' % re.sub("-", "", num2words[iN]), 'X[%s]' % iN, self.dF[iM])
                                #        self.dF[iM] = re.sub('log', 'sp.log', self.dF[iM])
                print ("    ...done")

                self.F = re.sub('log','sp.log',self.F)	

