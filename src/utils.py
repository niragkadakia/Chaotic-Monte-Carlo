"""
Helper functions

Created by Nirag Kadakia at 12:00 06-09-2018
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""


import scipy as sp


def diag_normal_pdf(x, variance, num_samples):
	""" 
	Sample from diagonal multivariate normal distribution with mean zero.
	"""
	
	if sp.array_equal(variance, sp.ones(sp.shape(x)[0])):
		pdf_tmp = sp.exp(-(x**2.0)/2.).T/(2.*sp.pi)**.5
	else:
		var_batch = (sp.ones((sp.shape(x)[0], num_samples)).T*variance).T
		pdf_tmp = sp.exp(-(x**2.0)/2./var_batch).T/(2.*sp.pi*variance)**.5
	return (sp.prod(pdf_tmp, axis=1))