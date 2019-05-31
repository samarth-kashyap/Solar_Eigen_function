import numpy as np
import functions as fn
import matplotlib.pyplot as plt
import scipy.integrate
import get_kernels as gkerns
plt.ion()

#timing the code
import timing
clock2 = timing.stopclock()
tstamp = clock2.lap

n,l = 1,3
n_,l_ = n,l

smax = 1

if(smax > 2*l+1) : exit()

s = np.arange(1,smax+1,1)  #confined to l_= l

#loading required functions
r = np.loadtxt('r.dat')
rpts = 700

r = r[-rpts:]

#for now considering l = l_
m = np.arange(-l,l+1,1)   # -l<=m<=l

#constructing meshgrid for feeding the kernel-finder
mm,mm_,ss1 = np.meshgrid(m,m,s,indexing='ij')

__,__,ss2,__ = np.meshgrid(m,m,s,r,indexing='ij')

kern_eval = gkerns.Hkernels(l,l_,ss1,ss2,r,rpts)

##computing the tensor components
#Bmm,B0m,B00,Bpm,Bpp,B0p = kern_eval.ret_kerns(l,ss1,l_,mm,mm_,n,n_)

Bmm,B0m,B00,Bpm,Bpp,B0p = kern_eval.isol_multiplet(n,l,s)

tstamp('kernel evaluated')
#sample h's for now

hmm = 100.*np.ones(np.shape(Bmm))
hpp = h00 = hpm = h0p = hmm
h0m = -259.

plt.plot(r,B0m[-2,-1,0,:])
plt.show('Block')

#find integrand by summing all component
cp_mat_s = hpp*Bpp + h00*B00 + hmm*Bmm \
           + 2*hpm*Bpm + 2*h0m*B0m + 2*h0p*B0p



#summing over s before carrying out radial integral
cp_mat_befint = np.sum(cp_mat_s,axis=2)

#radial integral
cp_mat = scipy.integrate.trapz(cp_mat_befint,x=r,axis=2)


plt.pcolormesh(cp_mat)
plt.colorbar()
plt.show('Block')
