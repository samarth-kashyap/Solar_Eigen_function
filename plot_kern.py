import numpy as np
import matplotlib.pyplot as plt
#from scipy import integrate
#from functions import * #importing file for evaluating derivative
import functions as fn
from os import getcwd

n,l,m = 1,60,0
n_,l_,m_ = n,l,m
nl = fn.find_nl(n,l)
nl_ = fn.find_nl(n_,l_)
s = 22
t = m_-m


if(nl == None or nl_ == None):
	print("Mode not found. Exiting."); exit()

#loading required functions
eig_dir = (getcwd() + '/eig_files')
U = fn.load_U(n,l,eig_dir)
V = fn.load_V(n,l,eig_dir)
U_ = fn.load_U(n_,l_,eig_dir)
V_ = fn.load_V(n_,l_,eig_dir)
r = np.loadtxt('r.dat')
rho = np.loadtxt('rho.dat')

#print integrate.simps(r*r*rho*(U_*U + l*(l+1.) * V_*V),r, even= 'avg')
#plt.plot(r,U*U,'r-')
#plt.plot(r,V*V,'b-')

#setting up shorthand repeatedly used in kernel evaluation
def wig_red(m1,m2,m3):
	'''3j symbol with upper row fixed'''
	return fn.wig(l_,s,l,m1,m2,m3)
#common prefactor appearing in all kernels
prefac = np.sqrt((2*l_+1.) * (2*s+1.) * (2*l+1.) / (4.* np.pi)) * wig_red(-m_,t,m)
dU,dV = fn.deriv(U,r), fn.deriv(V,r)
dU_,dV_ = fn.deriv(U_,r), fn.deriv(V_,r)
d2U_,d2V_ = fn.deriv(dU_,r), fn.deriv(dV_,r)
om = fn.omega


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
start_ind = fn.nearest_index(r,r_start)
plt.plot(r[start_ind:],(rho*B0m)[start_ind:],'k-')
plt.grid(True)
plt.show()

















