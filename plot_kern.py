import numpy as np
import matplotlib.pyplot as plt
#from scipy import integrate
from sympy.physics.wigner import wigner_3j
from sympy import N as sympy_eval
from functions import * #importing file for evaluating derivative
from os import getcwd

def wig(l1,l2,l3,m1,m2,m3):
	"""numerical value of wigner3j symbol"""
	return(sympy_eval(wigner_3j(l1,l2,l3,m1,m2,m3)))

def om(l,n):
	"""numerical value of \Omega_l^n"""
	return np.sqrt(0.5*(l+n)*(l-n+1.))

nl_list = np.loadtxt('nl.dat')

n,l,m = 1,60,0
n_,l_,m_ = n,l,m
nl = None
nl_ = None
s = 22
t = m_-m

for i in range(len(nl_list)):	
	if (np.array_equal(nl_list[i],np.array([n,l]))):	nl = i
	if (np.array_equal(nl_list[i],np.array([n_,l_]))):	nl_ = i
if(nl == None or nl_ == None):
	print("Mode not found. Exiting.")
	exit()

eig_dir = (getcwd() + '/eig_files')
U = np.loadtxt(eig_dir + '/'+'U'+str(nl)+'.dat')
V = np.loadtxt(eig_dir + '/'+'V'+str(nl)+'.dat')
U_ = np.loadtxt(eig_dir + '/'+'U'+str(nl_)+'.dat')
V_ = np.loadtxt(eig_dir + '/'+'V'+str(nl_)+'.dat')
r = np.loadtxt('r.dat')
rho = np.loadtxt('rho.dat')

#print integrate.simps(r*r*rho*(U_*U + l*(l+1.) * V_*V),r, even= 'avg')
#plt.plot(r,U*U,'r-')
#plt.plot(r,V*V,'b-')

#setting up shorthand repeatedly used in kernel evaluation
def wig_red(m1,m2,m3):
	'''3j symbol with upper row fixed'''
	return wig(l_,s,l,m1,m2,m3)
#common prefactor appearing in all kernels
prefac = np.sqrt((2*l_+1.) * (2*s+1.) * (2*l+1.) / (4.* np.pi)) * wig_red(-m_,t,m)
dU,dV = deriv(U,r), deriv(V,r)
dU_,dV_ = deriv(U_,r), deriv(V_,r)
d2U_,d2V_ = deriv(dU_,r), deriv(dV_,r)


#B-- EXPRESSION
Bmm = wig_red(3,-2,-1)*om(l,0)*om(l_,0)*om(l_,2)*om(l_,3) * V*V_
Bmm += wig_red(0,-2,2)*om(l,0)*om(l,2) * r*V*dU_
Bmm += wig_red(1,-2,1)*om(l_,0)*om(l,0) * (-U*U_ + U*V_ + om(l_,2)**2 * V*V_ - r*U*V_)
Bmm += wig_red(2,-2,0)*om(l_,0)*om(l_,2) * (U*V_ + r*dU*V_ - r*U*dV_)
Bmm *= (-1)**(1+m_)/(r*r) * prefac

#B0- EXPRESSION
B0m = wig_red(0,-1,1)*om(l,0) * (2*U*U_ + om(l_,2)**2*V*U_ + om(l_,0)**2*(-2*U*V_ + V*V_ + r*V*dV_) + r*(-U - V + r*dV)*dU_)
B0m += wig_red(-1,-1,2)*om(l,0)*om(l_,0)*om(l,2) * V * (U_ - V_ + r*V_)
B0m *=  0.5*(-1)**(m_)/(r*r) * prefac
B0m += wig_red(2,-1,-1)*om(l,0)*om(l_,0)*om(l_,2) * (V*U_ - 3*V*V_ + r*V*dV_ + 2*r*dV*V_)
B0m -= wig_red(1,-1,0)*om(l_,0) * (2*U*U_ + om(l_,0)**2*V*U_ + om(l,0)**2*(-V*V_ + r*V*V_) + U*(2*V_ + r*(dU_ - 2*dV_ + r*d2V_)))

r_start = 0.9
start_ind = nearest_index(r,r_start)
plt.plot(r[start_ind:],(rho*B0m)[start_ind:],'k-')
plt.grid(True)
plt.show()

















