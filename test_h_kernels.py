import numpy as np
import functions as fn
import matplotlib.pyplot as plt
import scipy.integrate
import get_kernels as gkerns
import h_components as hcomps
plt.ion()
#code snippet for timing the code
import timing
clock2 = timing.stopclock()
tstamp = clock2.lap

l=2
l_=2
s = np.array([0,1,2])
m = np.arange(-l,l+1,1)   # -l<=m<=l
m_ = np.arange(-l_,l_+1,1) # -l_<=m<=l_
l_b = 1
m_b = 0

r = np.loadtxt('r.dat')
r = r[-700:]
r_cen = np.mean(r)
b = lambda r: np.exp(-0.5*((r-r_cen)/0.0001)**2) 
b_r = b(r)

#Fetching the H-components
get_h = hcomps.getHcomps(s,m,l_b,m_b,r,b_r)

tstamp()
H_super = get_h.ret_hcomps()  #this is where its getting computed
tstamp('Computed H-components in')

#distributing the components
hmm = H_super[0,0,:,:,:,:]
h0m = H_super[1,0,:,:,:,:]
h00 = H_super[1,1,:,:,:,:]
hp0 = H_super[2,1,:,:,:,:]
hpp = H_super[2,2,:,:,:,:]
hpm = H_super[2,0,:,:,:,:]
