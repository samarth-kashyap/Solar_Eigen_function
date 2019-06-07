import numpy as np
import functions as fn
import timing
import get_kernels as gkerns
import matplotlib.pyplot as plt

s = np.array([1])
m = np.array([2])
m_ = np.array([2])
n,l = 1,2
n_,l_ = n,l
r = np.loadtxt('r.dat')
kern = gkerns.Hkernels(n_,l_,n,l,s,-700,-1)

Bmm, B0m,B00, Bpm,_,_ = kern.ret_kerns()
plt.plot(r[-700:-1],B0m[0,0,0,:])
plt.show()
