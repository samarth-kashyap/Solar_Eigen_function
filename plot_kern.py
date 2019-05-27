#Code reads eigenvalue functions and plots loentz stress kernels

#Defining function and variables for timing code. Not essential to working of code
from time import clock
start_time = clock() #ref time
last_time = start_time #ref time
def tstamp(stampname = None):
	'''returns time elapsed since beginning of runtime'''	
	this_time = clock()
	global last_time
	if(stampname != None):
		print(stampname + ': ' + str(this_time - last_time))
	last_time = this_time
	
#Beginning of main part
import numpy as np
import matplotlib.pyplot as plt
import functions as fn
from os import getcwd

tstamp('library loading') #printing elapsed time from beginning of runtime

n,l,m = 1,60,1
n_,l_,m_ = n,l,2
nl = fn.find_nl(n,l)
nl_ = fn.find_nl(n_,l_)
s = 22
t = m_-m


#Savitsky golay filter for smoothening
window = 45  #must be odd
order = 3

if(nl == None or nl_ == None):
	print("Mode not found. Exiting."); exit()

#loading required functions
eig_dir = (getcwd() + '/eig_files')
U,V = fn.load_eig(n,l,eig_dir)
U_,V_= fn.load_eig(n_,l_,eig_dir)
r = np.loadtxt('r.dat')
rho = np.loadtxt('rho.dat')

#interpolation params
npts = 30000
r_new = np.linspace(np.amin(r),np.amax(r),npts)

tstamp('files loading')

#print integrate.simps(r*r*rho*(U_*U + l*(l+1.) * V_*V),r, even= 'avg')
#plt.plot(r,U*U,'r-')
#plt.plot(r,V*V,'b-')

#setting up shorthand repeatedly used in kernel evaluation
def wig_red(m1,m2,m3):
	'''3j symbol with upper row fixed'''
	return fn.wig(l_,s,l,m1,m2,m3)
#common prefactor appearing in all kernels
prefac = np.sqrt((2*l_+1.) * (2*s+1.) * (2*l+1.) / (4.* np.pi)) * wig_red(-m_,t,m)
'''dU,dV = fn.deriv(U,r), fn.deriv(V,r)
dU_,dV_ = fn.deriv(U_,r), fn.deriv(V_,r)
dU, dV = fn.smooth(dU, 5), fn.smooth(dV,5)
dU_,dV_ = fn.smooth(dU_,7), fn.smooth(dV_,7)
d2U_,d2V_ = fn.deriv(dU_,r), fn.deriv(dV_,r)
d2U_,d2V_ = fn.smooth(d2U_,10), fn.smooth(d2V_,10)'''

U,dU,d2U = fn.smooth(U,r,window,order,npts)
V,dV,d2V = fn.smooth(V,r,window,order,npts)

U_,dU_,d2U_ = fn.smooth(U_,r,window,order,npts)
V_,dV_,d2V_ = fn.smooth(V_,r,window,order,npts)

rho_sm, __, __ = fn.smooth(rho,r,window,order,npts)


#d2U_, d2V_ = fn.deriv2(U_,r), fn.deriv2(V_,r)
om = fn.omega
p = (-1)**(l+l_+s) #parity of selected modes

#re-assigning with smoothened variables
r = r_new
rho = rho_sm

#B-- EXPRESSION
Bmm = wig_red(3,-2,-1)*om(l,0)*om(l_,0)*om(l_,2)*om(l_,3) * V*V_
Bmm += wig_red(0,-2,2)*om(l,0)*om(l,2) * r*V*dU_
Bmm += wig_red(1,-2,1)*om(l_,0)*om(l,0) * (-U*U_ + U*V_ + om(l_,2)**2 * V*V_ - r*U*V_)
Bmm += wig_red(2,-2,0)*om(l_,0)*om(l_,2) * (U*V_ + r*dU*V_ - r*U*dV_)
Bmm *= (-1)**(1+m_)/(r*r) * prefac

#B0- EXPRESSION
B0m = wig_red(0,-1,1)*om(l,0) * (2*U*U_ + om(l_,2)**2*V*U_ + om(l_,0)**2*(-2*U*V_ + V*V_ + r*V*dV_) + r*(-U - V + r*dV)*dU_)
B0m += wig_red(-1,-1,2)*om(l,0)*om(l_,0)*om(l,2) * V * (U_ - V_ + r*V_)
B0m += wig_red(2,-1,-1)*om(l,0)*om(l_,0)*om(l_,2) * (V*U_ - 3*V*V_ + r*V*dV_ + 2*r*dV*V_)
B0m -= wig_red(1,-1,0)*om(l_,0) * (2*U*U_ + om(l_,0)**2*V*U_ + om(l,0)**2*(-V*V_ + r*V*V_) + U*(2*V_ + r*(dU_ - 2*dV_ + r*d2V_)))
B0m *=  0.5*(-1)**(m_)/(r*r) * prefac

#B00 EXPRESSION
B00 = -(wig_red(-1,0,1)+wig_red(1,0,-1))*om(l_,0)*om(l,0) * (V*(-4*U_ + 2*(1+om(l_,0)**2)*V_ + r*(dU_ - 2*dV_)) + 2*r*dV*(U_ - V_ + r*dV_))
B00 += wig_red(0,0,0) * ((6*U - 4*om(l,0)**2*V -2*r*dU)*U_ + 2*om(l_,0)**2*((-3*U+2*om(l,0)**2*V + r*dU)*V_ + r*U*dV_) + r*((-4*U + 2*om(l,0)**2*V + r*dU)*U_ + r*U*d2U_))
B00 *= 0.5*(-1)**(m_)/(r*r) * prefac

#B+- EXPRESSION
Bpm = -2*(1+p)*wig_red(-2,0,-2)*om(l_,0)*om(l,0)*om(l_,2)*om(l,2)*V*V_
Bpm += (1+p)*wig_red(-1,0,1)*om(l_,0)*om(l,0) * (-r*V*dU_ + U*(U_-V_-r*dV_))
Bpm += wig_red(0,0,0)*r*r * (-dU*dU_ + U*d2U_)
Bpm *= 0.5*(-1)**(m_)/(r*r) * prefac

tstamp('calculations')

r_start = 0.9
start_ind = fn.nearest_index(r,r_start)
plt.plot(r[start_ind:],(rho*Bpm)[start_ind:],'g-')
plt.plot(r[start_ind:],(rho*Bmm)[start_ind:],'r-')
plt.plot(r[start_ind:],(rho*B0m)[start_ind:],'b-')
plt.plot(r[start_ind:],(rho*B00)[start_ind:],'k-')
plt.legend(['$\mathcal{B}^{+-}$','$\mathcal{B}^{--}$','$\mathcal{B}^{0-}$','$\mathcal{B}^{00}$'])
plt.grid(True)
plt.show()

tstamp('plotting')



