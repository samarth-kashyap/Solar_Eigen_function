# python library of functions used in Lorentz stress kernel evaluation

import numpy as np
from sympy.physics.wigner import wigner_3j
from sympy import N as sympy_eval

#evaluation
def wig(l1,l2,l3,m1,m2,m3):
	"""returns numerical value of wigner3j symbol"""
	return(sympy_eval(wigner_3j(l1,l2,l3,m1,m2,m3)))

def omega(l,n):
	"""returns numerical value of \Omega_l^n"""
	return np.sqrt(0.5*(l+n)*(l-n+1.))

def deriv(y,x):
	"""returns derivative of y wrt x. same len as x and y"""
	if(len(x) != len(y)):	
		print("lengths dont match")
		exit()
	l = len(y)	
	ret = np.zeros(l)
	ret[0] = (y[1]-y[0]) / (x[1]-x[0])
	ret[-1] = (y[-1]-y[-2]) / (x[-1]-x[-2])
	for i in range(1,l-1):
		ret[i] = (y[i+1]-y[i-1]) / (x[i+1]-x[i-1])
	return ret

def deriv2(y,x):
	"""returns second derivative of y wrt x"""
	if(len(x) != len(y)):	
		print("lengths dont match")
		exit()
	l = len(y)	
	ret = np.zeros(l)
	for i in range(1,l-1):
		xf,yf = x[i+1], y[i+1]
		xb,yb = x[i-1], y[i-1]
		xx,yy = x[i], y[i]
		ret[i] = 2./(xf-xb) * ((yf-yy)/(xf-xx) - (yy-yb)/(xx-xb))
	ret[0], ret[-1] = ret[1], ret[-2]
	return ret
	
def nearest_index(array, value):
	"""returns index of object nearest to value in array"""
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx

#loading
nl_list = np.loadtxt('nl.dat')	
def find_nl(n,l):
	"""returns nl index from given n and l"""
	for i in range(len(nl_list)):
		if (np.array_equal(nl_list[i],np.array([n,l]))):
			return i
	return None #when mode not found in nl_lsit

def find_mode(nl):
	"""returns (n,l) for given nl"""
	return int(nl_list[nl][0]), int(nl_list[nl][1])

def load_eig(n,l,eig_dir):
	"""returns U,V for mode n,l stored in directory eig_dir"""
	nl = find_nl(n,l)
	if (nl == None):
		print "mode doesn't exist in nl_list. exiting."
		exit()
	U = np.loadtxt(eig_dir + '/'+'U'+str(nl)+'.dat')
	V = np.loadtxt(eig_dir + '/'+'V'+str(nl)+'.dat')	
	return U,V
