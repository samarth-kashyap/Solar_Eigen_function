import numpy as np
import functions as fn
import matplotlib.pyplot as plt
import scipy.integrate
import get_kernels as gkerns
import h_components as hcomps
import sys
import math
import submatrix
plt.ion()
#code snippet for timing the code
import timing
clock2 = timing.stopclock()
tstamp = clock2.lap

OM = np.loadtxt('OM.dat')

r = np.loadtxt('r.dat')

field_type = 'dipolar'
        
n,n_ = 5,5
l=100
l_=l
s = np.array([0,1,2])
m = np.arange(-l,l+1,1)   # -l<=m<=l
m_ = np.arange(-l_,l_+1,1) # -l_<=m<=l_
s0 = 1
t0 = np.arange(-s0,s0+1)

eigvals = submatrix.lorentz_diagonal(n_,n,l_,l,r,field_type)
Lambda = np.diag(eigvals)

#mm_,mm = np.meshgrid(m_,m,indexing='ij')

#plt.pcolormesh(mm,mm_,np.real(Lambda))
#plt.colorbar()
#plt.gca().invert_yaxis()
#plt.show('Block')

omega = np.loadtxt('muhz.dat')
omega = omega[fn.find_nl(n,l)] * 1e-6 / OM
f_dpt = np.sqrt(omega**2 + eigvals) * 1e6 * OM
plt.plot(f_dpt,'.')
plt.plot(omega*1e6*OM*np.ones(len(f_dpt)))
plt.show()

#plt.subplot()
#plt.pcolormesh(np.imag(Lambda))
#plt.gca().invert_yaxis()
#plt.colorbar()
#plt.show('Block')
