import numpy as np
import functions as fn
import matplotlib.pyplot as plt
import scipy.integrate
import get_kernels as gkerns
plt.ion()
#code snippet for timing the code
import timing
clock2 = timing.stopclock()
tstamp = clock2.lap


#Choosing n,l and n_,l_
n,l = 1,10
n_,l_ = n,l

#Choosing s
smin = 0
#smax = 'max_s' for 0 <= s <= 2l+1
#smax = False for s = s_arr (custom choice)
#smax = <some integral value> 
smax = 'max_s'
#custom made s array
s_arr = np.array([1,2,3,4])


#assigning the s array according to user's choice
if(smax == 'max_s'): s = np.arange(0,2*l+1,1)  #to generate all s
elif(smax == False): s = s_arr  #to stick to a particular s_array
else: s = np.arange(smin,smax+1,1)  #generate s between a max and min value


#loading and chopping r grid
r = np.loadtxt('r.dat')
rpts = 700
r = r[-rpts:]


#for now considering l = l_
m = np.arange(-l,l+1,1)   # -l<=m<=l


#constructing meshgrid for feeding the generic kernel-finder
# mm,mm_,ss1 = np.meshgrid(m,m,s,indexing='ij')
# __,__,ss2,__ = np.meshgrid(m,m,s,r,indexing='ij')
# kern_eval = gkerns.Hkernels(l,l_,r,rpts,ss1,ss2)
##computing the tensor components
#Bmm,B0m,B00,Bpm,Bpp,B0p = kern_eval.ret_kerns(l,ss1,l_,mm,mm_,n,n_)


#For the current case where l=l_,n=n_
kern_eval = gkerns.Hkernels(l,l_,r,rpts)

Bmm,B0m,B00,Bpm,Bpp,B0p = kern_eval.isol_multiplet(n,l,s)

tstamp('kernel evaluated')

#sample h's for now
hmm = 100.*np.ones(np.shape(Bmm))
hpp = h00 = hpm = h0p = hmm
h0m = -259.


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
