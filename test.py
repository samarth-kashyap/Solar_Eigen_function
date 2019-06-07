import numpy as np
import functions as fn
import timing
import get_kernels as gkerns
import matplotlib.pyplot as plt

s = np.array([1])
m = np.array([2])
m_ = np.array([2])
l = 2
r = np.loadtxt('r.dat')
kern = gkerns.Hkernels(l,l,r,700)

mm, mm_, ss = np.meshgrid(m,m_,s,indexing='ij')

Bmm, B0m,B00, Bpm,_,_ = kern.ret_kerns(l,ss,l,mm,mm_,1,1)
plt.plot(r[-700:],B0m[0,0,0,:])
print 'lol'
plt.show()
