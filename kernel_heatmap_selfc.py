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

r_start = 0.68
r_end = 1.0
start_ind = fn.nearest_index(r,r_start)
end_ind = fn.nearest_index(r,r_end)  #ends at the surface
r = r[start_ind:end_ind+1]
OM = np.loadtxt('OM.dat')

omega_list = np.loadtxt('muhz.dat')

#self coupling
m = np.array([0])
m_ = np.array([0])
s = np.array([8])

nl_list = np.loadtxt('nl.dat')
nl_arr = nl_list[nl_list[:,0]==0]
nl_arr = nl_arr[nl_arr[:,1]>2]

multiplyrho = True

plot_fac = OM**2 * 1e12 * 1e-10 #unit = muHz^2 G^(-2) V_sol^(-1)

#rearranging so that we can iterate for all l's over a fixed n's
for i in range(1,33):   #max n goes upto 32
    nl_arr_temp = nl_list[nl_list[:,0]==i]
    nl_arr = np.append(nl_arr,nl_arr_temp[nl_arr_temp[:,1]>2],axis=0)


nl_arr = nl_arr.astype(int)

Bmm_all = np.zeros((len(nl_arr),npts))
B0m_all = np.zeros((len(nl_arr),npts))
B00_all = np.zeros((len(nl_arr),npts))
Bpm_all = np.zeros((len(nl_arr),npts))

#some dummy n,l to get rho. I know its unclean.
n0 = nl_arr[0,0]
l0 = nl_arr[0,1]
omega_nl0 = omega_list[fn.find_nl(n0, l0)]

rho,__,__,__,__,_,_ = np.array(gkerns.Hkernels(n0,l0,m,n0,l0,m,s,r)\
                        .ret_kerns_axis_symm(a_coeffkerns = True))

for i in range(len(nl_arr)):
    nl = nl_arr[i]
    n = nl[0]
    l = nl[1]

    omega_nl = omega_list[fn.find_nl(n, l)]

    print(s,n,l)
    __, Bmm, B0m,B00, Bpm,_,_ = np.array(gkerns.Hkernels(n,l,m,n,l,m,s,r)\
                        .ret_kerns_axis_symm(a_coeffkerns = True))*plot_fac/(-2*omega_nl)

    Bmm_all[i,:] = Bmm
    B0m_all[i,:] = B0m
    B00_all[i,:] = B00
    Bpm_all[i,:] = Bpm

if(multiplyrho==True):
    Bmm_all, B0m_all,B00_all, Bpm_all = rho[np.newaxis,:]*Bmm_all, \
        rho[np.newaxis,:]*B0m_all,rho[np.newaxis,:]*B00_all, rho[np.newaxis,:]*Bpm_all

np.savetxt('./kernels_self/Bmm_all.dat',Bmm_all)
np.savetxt('./kernels_self/B0m_all.dat',B0m_all)
np.savetxt('./kernels_self/B00_all.dat',B00_all)
np.savetxt('./kernels_self/Bpm_all.dat',Bpm_all)


plt.pcolormesh(Bmm_all)