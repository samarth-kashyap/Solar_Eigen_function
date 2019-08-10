import timing
import numpy as np
import matplotlib.pyplot as plt
import functions as fn
from os import getcwd
import get_kernels as gkerns

kernclock = timing.stopclock()
tstamp = kernclock.lap

r = np.loadtxt('r.dat')
start_ind = fn.nearest_index(r,0.5)
end_ind = fn.nearest_index(r,1.)
r = r[start_ind:end_ind+1]
OM = np.loadtxt('OM.dat')

n1,l1 = 4,3
n2,l2 = 1,10
n3,l3 = 0,60

m = np.array([0])
m_ = np.array([0])
s = np.array([2])

plot_fac = OM**2 * 1e12 * (4.*np.pi/3) * 1e-10 #unit = muHz^2 G^(-2) V_sol^(-1)

#kern = gkerns.Hkernels(n1,l1,m,n1,l1,m,s,r)
Bmm1, B0m1,B001, Bpm1,_,_ = np.array(gkerns.Hkernels(n1,l1,m,n1,l1,m,s,r).ret_kerns_axis_symm())*plot_fac
Bmm2, B0m2,B002, Bpm2,_,_ = np.array(gkerns.Hkernels(n2,l2,m,n2,l2,m,s,r).ret_kerns_axis_symm())*plot_fac
Bmm3, B0m3,B003, Bpm3,_,_ = np.array(gkerns.Hkernels(n3,l3,m,n3,l3,m,s,r).ret_kerns_axis_symm())*plot_fac
tstamp('kernel calculation time')

npts = 100   #check the npts in get_kernels
r_new = np.linspace(np.amin(r),np.amax(r),npts)
r = r_new

dpi = 80
plt.figure(num=None, figsize=(10, 8), dpi=dpi, facecolor='w', edgecolor='k')

plt.subplot(221)
plt.plot(r,Bmm1[0,0], label = '('+str(n1)+','+str(l1)+')')
plt.plot(r,Bmm2[0,0], label = '('+str(n2)+','+str(l2)+')')
plt.plot(r,Bmm3[0,0], label = '('+str(n3)+','+str(l3)+')')
plt.xlabel('$r/R_{\odot}$')
plt.legend()
plt.grid(True)
plt.ylabel('$\mathcal{B}^{--}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')

plt.subplot(222)
plt.plot(r,B0m1[0,0], label = '('+str(n1)+','+str(l1)+')')
plt.plot(r,B0m2[0,0], label = '('+str(n2)+','+str(l2)+')')
plt.plot(r,B0m3[0,0], label = '('+str(n3)+','+str(l3)+')')
plt.xlabel('$r/R_{\odot}$')
plt.legend()
plt.grid(True)
plt.ylabel('$\mathcal{B}^{0-}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')

plt.subplot(223)
plt.plot(r,B001[0,0], label = '('+str(n1)+','+str(l1)+')')
plt.plot(r,B002[0,0], label = '('+str(n2)+','+str(l2)+')')
plt.plot(r,B003[0,0], label = '('+str(n3)+','+str(l3)+')')
plt.xlabel('$r/R_{\odot}$')
plt.legend()
plt.grid(True)
plt.ylabel('$\mathcal{B}^{00}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')

plt.subplot(224)
plt.plot(r,Bpm1[0,0], label = '('+str(n1)+','+str(l1)+')')
plt.plot(r,Bpm2[0,0], label = '('+str(n2)+','+str(l2)+')')
plt.plot(r,Bpm3[0,0], label = '('+str(n3)+','+str(l3)+')')
plt.xlabel('$r/R_{\odot}$')
plt.legend()
plt.grid(True)
plt.ylabel('$\mathcal{B}^{+-}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.savefig('./kern2.pdf')

plt.show()
