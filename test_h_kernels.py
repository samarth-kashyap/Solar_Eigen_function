import numpy as np
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

#all quantities in cgs
M_sol = 1.989e33 #g
R_sol = 6.956e10 #cm
B_0 = 10e5 #G
OM = np.sqrt(4*np.pi*R_sol*B_0**2/M_sol)

r = np.loadtxt('r.dat')
r_start, r_end = 0.6, 0.61
start_ind, end_ind = [fn.nearest_index(r, pt) for pt in (r_start, r_end)]
#end_ind = start_ind + 700
r = r[start_ind:end_ind]

#transition radii for mixed field type
R1 = 0.61
R2 = 0.62

field_type = 'dipolar'
        
n,n_ = 2,2
l=5
l_=l
s = np.array([0,1,2])
m = np.arange(-l,l+1,1)   # -l<=m<=l
m_ = np.arange(-l_,l_+1,1) # -l_<=m<=l_
s0 = 1
t0 = np.arange(-s0,s0+1)

r_cen = np.mean(r)
                                        
B_mu_t_r = fn.getB_comps(s0,r,R1,R2,start_ind,end_ind,field_type)

get_h = hcomps.getHcomps(s,m,s0,t0,r,B_mu_t_r, 0.)

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

