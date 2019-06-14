import timing
import numpy as np
import matplotlib.pyplot as plt
import functions as fn
from os import getcwd
import get_kernels as gkerns
import get_kernels_separate as gkerns_sep

kernclock = timing.stopclock()
tstamp = kernclock.lap

r = np.loadtxt('r.dat')
r_start, r_end = 0.9, 0.98
start_ind, end_ind = [fn.nearest_index(r, pt) for pt in (r_start, r_end)]
#end_ind = start_ind + 700
r = r[start_ind:end_ind]

n,l = 1,20
n_,l_ = n,l
m = np.array([2])
m_ = np.array([2])
s = np.array([22])

kern = gkerns.Hkernels(n_,l_,m_,n,l,m,s,start_ind,end_ind)
Bmm, B0m,B00, Bpm,_,_ = kern.ret_kerns()
tstamp('kernel calculation time')
kern1 = gkerns_sep.Hkernels(n_,l_,m_,n,l,m,s,start_ind,end_ind)
Bmm1, B0m1,B001, Bpm1,_,_ = kern1.ret_kerns()
tstamp('kernel calculation time separate')

plt.subplot(221)
plt.plot(r,np.abs(Bpm-Bpm1)[0,0,0],'r--',label = '$\mathcal{B}^{+-}$')
plt.plot(r,Bpm[0,0,0],'r-',label = '$\mathcal{B}^{+-}$')
plt.grid(True)
plt.legend()

plt.subplot(222)
plt.plot(r,np.abs(Bmm-Bmm1)[0,0,0],'b--',label = '$\mathcal{B}^{--}$')
plt.plot(r,Bmm[0,0,0],'b-',label = '$\mathcal{B}^{--}$')
plt.grid(True)
plt.legend()

plt.subplot(223)
plt.plot(r,np.abs(B0m-B0m1)[0,0,0],'k--',label = '$\mathcal{B}^{0-}$')
plt.plot(r,B0m[0,0,0],'k-',label = '$\mathcal{B}^{0-}$')
plt.grid(True)
plt.legend()

plt.subplot(224)
plt.plot(r,np.abs(B00-B001)[0,0,0],'g--',label = '$\mathcal{B}^{00}$')	
plt.plot(r,B00[0,0,0],'g-',label = '$\mathcal{B}^{00}$')	
plt.grid(True)
plt.legend()

plt.show()
