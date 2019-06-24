import numpy as np
import math
import functions as fn
import matplotlib.pyplot as plt
import scipy.integrate
import get_kernels as gkerns
import h_components as hcomps
import sys
import math
plt.ion()
#code snippet for timing the code
import timing
clock2 = timing.stopclock()
tstamp = clock2.lap
nperf = np.vectorize(math.erf)

r = np.loadtxt('r.dat')
r_start, r_end = 0.5, 0.9
start_ind, end_ind = [fn.nearest_index(r, pt) for pt in (r_start, r_end)]
#end_ind = start_ind + 700
r = r[start_ind:end_ind]

nperf = np.vectorize(math.erf)

#all quantities in cgs
M_sol = 1.989e33 #g
R_sol = 6.956e10 #cm
B_0 = 10e5 #G
OM = np.sqrt(4*np.pi*R_sol*B_0**2/M_sol)

field_type = 'dipolar'

if(field_type=='mixed'):
        R1 = 0.6
        R2 = 0.7
        R1_ind = np.argmin(np.abs(r-R1))
        R2_ind = np.argmin(np.abs(r-R2))
        b = 0.5*(1+nperf(70*(r-(R1+R2)/2.0)))
        a = b - np.gradient(b)*r
        
n,n_ = 2,2
l=5
l_=l
s = np.array([0,1,2])
m = np.arange(-l,l+1,1)   # -l<=m<=l
m_ = np.arange(-l_,l_+1,1) # -l_<=m<=l_
s0 = 1
t0 = np.arange(-s0,s0+1)

r_cen = np.mean(r) 
beta = lambda r: 1./r**3  #a global dipole
alpha = beta  #for now keeping the radial dependence same as dipolar

if(field_type=='mixed'):
        alpha = lambda r: np.exp(-0.5*(r/0.18)**2)  #goes to zero around r = 0.6 R_sun
B_mu_t_r = np.zeros((3,2*s0+1,len(r)),dtype=complex)

if(field_type == 'dipolar'):
        B_mu_t_r[:,s0,:] = 1e-4 * fn.omega(s0,0) * 1./np.sqrt(2.) \
                                * np.outer(np.array([1., -2., 1.]),beta(r))
elif(field_type == 'toroidal'):
        B_mu_t_r[:,s0,:] = 1e-4 * fn.omega(s0,0) * 1./np.sqrt(2.) \
                                * np.outer(np.array([-1j, 0. , 1j]),alpha(r))
else:
        B_mu_t_r[:,s0,start_ind:R1_ind] = 1e-4 * fn.omega(s0,0) * 1./np.sqrt(2.) \
                                * np.outer(np.array([-1j, 0. , 1j]),\
                                        alpha(r[start_ind:R1_ind]))
        B_mu_t_r[:,s0,R2_ind:end_ind] = 1e-4 * fn.omega(s0,0) * 1./np.sqrt(2.) \
                                * np.outer(np.array([1., -2., 1.]),\
                                        beta(r[R2_ind:end_ind]))
        B_mu_t_r[:,s0,R1_ind:R2_ind] = 1e-4 * fn.omega(s0,0) * 1./np.sqrt(2.) \
                                * np.array([1., -2., 1.])[:,np.newaxis]*\
                                        beta(r[R1_ind:R2_ind])*np.array([a[R1_ind:R2_ind],\
                                        b[R1_ind:R2_ind],a[R1_ind:R2_ind]])
        B_mu_t_r[:,s0,R1_ind:R2_ind] += 1e-4 * fn.omega(s0,0) * 1./np.sqrt(2.) \
                                * np.outer(np.array([-1j, 0., 1j]),\
                                        alpha(r[R1_ind:R2_ind]))



#B_mu_t_r = np.zeros((3,2*s0+1,len(r)))
#B_mu_t_r[:,s0,:] = 1e-4 * fn.omega(s0,0) * 1./np.sqrt(2.) * np.outer(np.array([1., -2., 1.]),beta(r))
#B_mu_t_r[:,0,:] = 1e-3 * fn.omega(s0,0) * 1./np.sqrt(2.) * np.outer(np.array([1., -2., 1.]),b(r))
#B_mu_t_r[:,2,:] = 1e-3 * fn.omega(s0,0) * 1./np.sqrt(2.) * np.outer(np.array([1., -2., 1.]),b(r))

#Fetching the H-components
get_h = hcomps.getHcomps(s,m,s0,t0,r,B_mu_t_r, 30.)

tstamp()

H_super = get_h.ret_hcomps()  #- sign due to i in B

###################TESTING
##H_super = np.transpose(H_super)
#eps = 1e-4
#theta = np.linspace(eps,np.pi-eps,20)
#rr,tt = np.meshgrid(r,theta,indexing='ij')
#xx = rr*np.cos(tt)
#yy = rr*np.sin(tt)

#print H_super.shape
#H_super = H_super.astype('complex128')
#H_superR = np.real(H_super)
#plt.pcolormesh(xx,yy,H_superR[:,:,0])
#plt.colorbar()


#sys.exit()
###################TESTING



tstamp('Computed H-components in')

#distributing the components
hmm = H_super[0,0,:,:,:,:]
h0m = H_super[1,0,:,:,:,:]
h00 = H_super[1,1,:,:,:,:]
hpm = H_super[2,0,:,:,:,:]
hp0 = H_super[2,1,:,:,:,:]
hpp = H_super[2,2,:,:,:,:]

kern = gkerns.Hkernels(n_,l_,m_,n,l,m,s,start_ind,end_ind)
Bmm,B0m,B00,Bpm,Bp0,Bpp = kern.ret_kerns()

#find integrand by summing all component
Lambda_sr = hpp*Bpp + h00*B00 + hmm*Bmm \
        + 2*hpm*Bpm + 2*h0m*B0m + 2*hp0*Bp0

#summing over s before carrying out radial integral
Lambda_r = np.sum(Lambda_sr,axis=2)

#radial integral
Lambda = scipy.integrate.trapz(Lambda_r*(r**2)[np.newaxis,:],x=r,axis=2)


#Lambda = np.real(Lambda)
Lambda *= OM**2 / 1e-3

eigenvalues,_ = np.linalg.eig(Lambda)
eigenvalues = np.sort(eigenvalues)
print eigenvalues

Lambda = np.transpose(Lambda)
Lambda = np.flip(Lambda, axis = 1)
plt.pcolormesh(np.real(Lambda)  )
plt.colorbar()
plt.show('Block')

#plt.subplot()
#plt.pcolormesh(np.imag(Lambda))
#plt.colorbar()
#plt.show('Block')

