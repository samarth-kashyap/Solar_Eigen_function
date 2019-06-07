import numpy as np
import functions as fn
import timing
import get_kernels as gkerns
import matplotlib.pyplot as plt

clock1 = timing.stopclock()
tstamp = clock1.lap

r = np.loadtxt('r.dat')
r_start, r_end = 0.9, 1.0
start_ind, end_ind = [fn.nearest_index(r, pt) for pt in (r_start, r_end)]
end_ind = start_ind + 700
r = r[start_ind:end_ind]

#n,l = 1,2
#n_,l_ = n,l
#m = np.array([2])
#m_ = np.array([2])
#s = np.array([1])

n,l = 1,60
n_,l_ = n,l
m = np.array([2])
m_ = np.array([2])
s = np.array([22])


kern = gkerns.Hkernels(n_,l_,m_,n,l,m,s,start_ind,end_ind)
Bmm, B0m,B00, Bpm,_,_ = kern.ret_kerns()
tstamp('kern time')
plt.plot(r,B0m[0,0,0,:])
plt.show()
