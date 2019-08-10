import timing
import numpy as np
import matplotlib.pyplot as plt
import functions as fn
from os import getcwd
import get_kernels as gkerns
import sys

kernclock = timing.stopclock()
tstamp = kernclock.lap

r = np.loadtxt('r.dat')
npts = 300

#starting from r = 0.5 R_sol
start_ind = fn.nearest_index(r,0.5)
end_ind = fn.nearest_index(r,1.)  #ends at the surface
r = r[start_ind:end_ind+1]
OM = np.loadtxt('OM.dat')

#self coupling
m = np.array([0])
m_ = np.array([0])
s = np.array([2])

nl_list = np.loadtxt('nl.dat')
nl_arr = nl_list[nl_list[:,0]==0]
nl_arr = nl_arr[nl_arr[:,1]>s[0]]

plot_fac = OM**2 * 1e12 * (4.*np.pi/3) * 1e-10 #unit = muHz^2 G^(-2) V_sol^(-1)

#rearranging so that we can iterate for all l's over a fixed n's
for i in range(1,33):   #max n goes upto 32
    nl_arr_temp = nl_list[nl_list[:,0]==i]
    nl_arr = np.append(nl_arr,nl_arr_temp[nl_arr_temp[:,1]>s[0]],axis=0)


nl_arr = nl_arr.astype(int)

Bmm_all = np.zeros((len(nl_arr),npts))
B0m_all = np.zeros((len(nl_arr),npts))
B00_all = np.zeros((len(nl_arr),npts))
Bpm_all = np.zeros((len(nl_arr),npts))

for i in range(len(nl_arr)):
    nl = nl_arr[i]
    n = nl[0]
    l = nl[1]
    print(n,l)
    Bmm, B0m,B00, Bpm,_,_ = np.array(gkerns.Hkernels(n,l,m,n,l,m,s,r)\
                                .ret_kerns_axis_symm())*plot_fac

    Bmm_all[i,:] = Bmm
    B0m_all[i,:] = B0m
    B00_all[i,:] = B00
    Bpm_all[i,:] = Bpm

np.savetxt('./kernels/Bmm_all.dat',Bmm_all)
np.savetxt('./kernels/B0m_all.dat',B0m_all)
np.savetxt('./kernels/B00_all.dat',B00_all)
np.savetxt('./kernels/Bpm_all.dat',Bpm_all)


plt.pcolormesh(Bmm_all)