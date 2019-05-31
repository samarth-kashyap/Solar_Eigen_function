import numpy as np
import functions as fn
import matplotlib.pyplot as plt
import scipy.integrate
import get_kernels as gkerns
plt.ion()
import timing
kernclock = timing.stopclock() #stopclock object for timing program
tstamp = kernclock.lap
n,l = 1,5
n_,l_ = n,l
tstamp('start')

s = np.arange(0,2*l+1,1)  #confined to l_= l

#loading required functions
r = np.loadtxt('r.dat')

#r = r[-100:]

#for now considering l = l_
m = np.arange(-l,l+1,1)   # -l<=m<=l

#constructing meshgrid for feeding the kernel-finder
tstamp()
mm,mm_,ss,__ = np.meshgrid(m,m,s,r,indexing='ij')
tstamp('meshgrid')
kern_eval = gkerns.Hkernels(l,l_,ss)
tstamp('object creation')
##computing the tensor components

Bmm,B0m,B00,Bpm,Bpp,B0p = kern_eval.ret_kerns(l,ss,l_,mm,mm_,n,n_)
tstamp('ret_kern')
#sample h's for now

hmm = 100*np.ones(np.shape(Bmm))
hpp = h00 = hpm = h0m = h0p = hmm

print(np.shape(Bmm))

#find integrand by summing all component
tstamp()
cp_mat_s = hpp*Bpp + h00*B00 + hmm*Bmm \
           + 2*hpm*Bpm + 2*h0m*B0m + 2*h0p*B0p
tstamp('contraction')
print(np.shape(cp_mat_s))

#summing over s before carrying out radial integral
cp_mat_befint = np.sum(cp_mat_s,axis=0)

print(np.shape(cp_mat_befint))

#radial integral
cp_mat = scipy.integrate.trapz(cp_mat_befint,x=r,axis=2)

print(np.shape(cp_mat))

plt.pcolormesh(cp_mat)
plt.colorbar()
