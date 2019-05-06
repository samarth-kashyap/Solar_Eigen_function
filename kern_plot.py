import numpy as np
import matplotlib.pyplot as plt
#from scipy import integrate
from sympy.physics.wigner import wigner_3j
from sympy import N as sympy_eval
#import deriv

def wig(l1,l2,l3,m1,m2,m3):
	"""numerical value of wigner3j symbol"""
	return(sympy_eval(wigner_3j(l1,l2,l3,m1,m2,m3)))

def om(l,n):
	"""numerical value of \Omega_l^n"""
	return np.sqrt(0.5*(l+n)*(l-n+1.))

nl_list = np.loadtxt('nl.dat')

n,l,m = 4,3,1
n_,l_,m_ = n,l,m
nl = None
nl_ = None
s = 5
t = m_-m

for i in range(len(nl_list)):	
	if (np.array_equal(nl_list[i],np.array([n,l]))):	nl = i
	if (np.array_equal(nl_list[i],np.array([n_,l_]))):	nl_ = i
if(nl == None or nl_ == None):
	print("Mode not found. Exiting.")
	exit()

U, V = np.loadtxt('eigU.dat')[nl], np.loadtxt('eigV.dat')[nl]
U_,V_ = np.loadtxt('eigU.dat')[nl_], np.loadtxt('eigV.dat')[nl_]
r = np.loadtxt('r.dat')
rho = np.loadtxt('rho.dat')

#print integrate.simps(r*r*rho*(U_*U + l*(l+1.) * V_*V),r, even= 'avg')
#plt.plot(r,U*U,'r-')
#plt.plot(r,V*V,'b-')

#setting up shorthands repeatedly used in kernel evaluation
def wig_red(m1,m2,m3):
	'''reduced form of wig'''
	return wig(l_,s,l,m1,m2,m3)
prefac = np.sqrt((2*l_+1.) * (2*s+1.) * (2*l+1.) / (4.* np.pi)) * wig_red(-m_,t,m)


#KERNEL EXPRESSIONS
Bmm = (-1)**(1+m)/(r*r) * prefac * (wig_red(3,-2,-1) * om(l,0) * (om(l_,0) * om(l_,2) * om(l_,3) * V * V_))
#print prefac,wig_red(3,-2,-1)
plt.plot(r[-2000:],Bmm[-2000:],'b-')
plt.show()






















