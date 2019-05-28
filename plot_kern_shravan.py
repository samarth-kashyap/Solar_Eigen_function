#PLOTTING SHRAVAN'S KERNEL (WITHOUT TERM6 AND TERM7)

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

n,l,m = 1,60,0
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

#setting up shorthand repeatedly used in kernel evaluation
def wig_red(m1,m2,m3):
	'''3j symbol with upper row fixed'''
	return fn.wig(l_,s,l,m1,m2,m3)
om = fn.omega
p = (-1)**(l+l_+s) #parity of selected modes
prefac = np.sqrt((2*l_+1.) * (2*s+1.) * (2*l+1.) / (4.* np.pi)) * wig_red(-m_,t,m)

#EIGENFUCNTION DERIVATIVES

#smoothing
U,dU,d2U = fn.smooth(U,r,window,order,npts)
V,dV,d2V = fn.smooth(V,r,window,order,npts)

U_,dU_,d2U_ = fn.smooth(U_,r,window,order,npts)
V_,dV_,d2V_ = fn.smooth(V_,r,window,order,npts)

rho_sm, __, __ = fn.smooth(rho,r,window,order,npts)
#re-assigning with smoothened variables
r = r_new
rho = rho_sm

##no smoothing
#dU, dV = np.gradient(U,r), np.gradient(V,r)
#dU_, dV_ = np.gradient(U_,r), np.gradient(V_,r)
#d2U_,d2V = 0.,0.


#B-- EXPRESSION
Bmm = -r*(wig_red(0,-2,2)*om(l,0)*om(l,2)*V*dU_ + wig_red(2,-2,0)*om(l_,0)* \
		om(l_,2)*V_*dU)
Bmm += wig_red(1,-2,1)*om(l_,0)*om(l,0)*(U-V)*(U_ - V_ + r*dV_)
Bmm *= ((-1)**m_)*prefac/(r**2)

#B-- EXTRA
Bmm_ = om(l_,0)*(wig_red(2,-2,0)*om(l_,2)*U*(V_ - r*dV_) + om(l,0)*V \
		*(wig_red(3,-2,-1)*om(l_,2)*om(l_,3)*V_ + wig_red(1,-2,1) \
		*(-U_ + V_ + om(l_,2)**2 *V_ - r*dV_)))
Bmm_ *= (-1)**(1+m_) *prefac/r**2

#B0- EXPRESSION
B0m = wig_red(1,-1,0)*om(l_,0)*(U - (om(l,0)**2)*V)*(U_ - V_ + r*dV_)
B0m += om(l,0)*(om(l_,0)*(wig_red(-1,-1,2)*om(l,2)*V*(U_ - V_ + r*dV_) \
       + 2*r*wig_red(2,-1,-1)*om(l_,2)*V_*dV) + wig_red(0,-1,1) \
	   *((U-V)*(2*U_ - 2*(om(l_,0)**2)*V_ - r*dU_) + r**2 * dU_*dV))
B0m *= 0.5*((-1)**m_)*prefac/r**2
#B0- EXTRA
B0m_ = om(l,0)*V*(wig_red(2,-1,-1)*om(l_,0)*om(l_,2)*(U_ - 3*V_ + r*dV_) \
		+ wig_red(0,-1,1)*((2+om(l_,0)**2)*U_ - 2*r*dU_ + om(l_,0)**2 \
		*(-3*V_ + r*dV_)))
B0m_ += wig_red(1,-1,0)*om(l_,0)*U*(U_ - V_ - r*(dU_ - dV_ + r*d2V_))
B0m_ *= 0.5*((-1)**m_)*prefac/r**2

#B00 OLD
B00 = -wig_red(0,0,0)*(2*U_ - 2*om(l_,0)**2 * V_ - r*dU_)*(-2*U + 2*om(l,0)**2 *V + \
		r*dU)
B00 -= 2*r*(wig_red(-1,0,1) + wig_red(1,0,-1))*om(l_,0)*om(l,0) \
       *(U_ - V_ + r*dV_)*dV
B00 *= 0.5*((-1)**m_)*prefac/r**2
#B00 EXTRA
B00_ = -(wig_red(-1,0,1) + wig_red(1,0,-1)) * om(l_,0)*om(l,0) * V*(-4*U_+2*(1+om(l_,0)**2)*V_+r*(dU_-2*dV_))
B00_ += wig_red(0,0,0)*U*(2*U_-2*r*dU_-2*om(l_,0)**2 *(V_-r*dV_)+r*r*d2U_)
B00_ *= 0.5*((-1)**m_)*prefac/r**2

#B+- OLD
Bpm = -r**2 * wig_red(0,0,0)*dU_*dU 
Bpm += om(l_,0)*om(l,0)*(-2*(wig_red(-2,0,2)+wig_red(2,0,-2))*om(l_,2)*om(l,2)*V_*V \
		+ wig_red(-1,0,1)*(U-V)*(U_ - V_ + r*dV_) + wig_red(1,0,-1) \
		*(U-V)*(U_ - V_ + r*dV_))
Bpm *= 0.5*((-1)**m_)*prefac/r**2
#B0+- EXTRA
Bpm_ = (wig_red(-1,0,1) + wig_red(1,0,-1)) * om(l_,0)*om(l,0) * V * (U_-V_+r*(-dU_+dV_))
Bpm_ += wig_red(0,0,0) * r*r*U*d2U_
Bpm_ *= 0.5*((-1)**m_)*prefac/r**2


tstamp('calculations')

r_start = 0.9
start_ind = fn.nearest_index(r,r_start)

plt.subplot(2,2,1)
plt.plot(r[start_ind:],(2*rho*Bpm)[start_ind:],'k-.')
plt.plot(r[start_ind:],(2*rho*Bpm_)[start_ind:],'r--')
plt.plot(r[start_ind:],(2*rho*(Bpm+Bpm_))[start_ind:],'b-')
plt.title('$\mathcal{B}^{+-}$')

plt.subplot(2,2,2)
plt.plot(r[start_ind:],(2*rho*Bmm)[start_ind:],'k-.')
plt.plot(r[start_ind:],(2*rho*Bmm_)[start_ind:],'r--')
plt.plot(r[start_ind:],(2*rho*(Bmm+Bmm_))[start_ind:],'b-')
plt.title('$\mathcal{B}^{--}$')

plt.subplot(2,2,3)
plt.plot(r[start_ind:],(2*rho*B0m)[start_ind:],'k-.')
plt.plot(r[start_ind:],(2*rho*B0m_)[start_ind:],'r--')
plt.plot(r[start_ind:],(2*rho*(B0m+B0m_))[start_ind:],'b-')
plt.title('$\mathcal{B}^{0-}$')

plt.subplot(2,2,4)
plt.plot(r[start_ind:],(2*rho*B00)[start_ind:],'k-.')
plt.plot(r[start_ind:],(2*rho*B00_)[start_ind:],'r--')
plt.plot(r[start_ind:],(2*rho*(B00+B00_))[start_ind:],'b-')
plt.title('$\mathcal{B}^{00}$')


#plt.plot(r[start_ind:],(2*rho*Bpm)[start_ind:],'k-',label = '$\mathcal{B}^{+-}$')
##plt.plot(r[start_ind:],(rho*Bmm)[start_ind:],'r-',label = '$\mathcal{B}^{--}$')
##plt.plot(r[start_ind:],(2*rho*B0m)[start_ind:],'k-',label = '$\mathcal{B}^{0-}$')
#plt.plot(r[start_ind:],(-rho*B00)[start_ind:],'k--',label = '$\mathcal{B}^{00}$')
#plt.grid(True)
plt.legend()
plt.show()

tstamp('plotting')





