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

#starting from r = 0.8 R_sol
start_ind = fn.nearest_index(r,0.8)
end_ind = fn.nearest_index(r,1.)  #ends at the surface
r = r[start_ind:end_ind+1]
OM = np.loadtxt('OM.dat')

plot_fac = OM**2 * 1e12 * (4.*np.pi/3) * 1e-10 #unit = muHz^2 G^(-2) V_sol^(-1)

#self coupling
m = np.array([0])
m_ = np.array([0])
s = np.array([50])

n0 = 1
l0 = 60

nl_list = np.loadtxt('nl.dat')
nl_arr = nl_list[nl_list[:,0]==0]
nl_arr = nl_arr[nl_arr[:,1]>3]


#rearranging so that we can iterate for all l's over a fixed n's
for i in range(1,33):   #max n goes upto 32
    nl_arr_temp = nl_list[nl_list[:,0]==i]
    nl_arr = np.append(nl_arr,nl_arr_temp[nl_arr_temp[:,1]>3],axis=0)



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

    if(np.abs(l0-l)>s[0]): continue

    Bmm, B0m,B00, Bpm,_,_ = np.array(gkerns.Hkernels(n0,l0,m,n,l,m,s,r)\
                                .ret_kerns_axis_symm())*plot_fac

    Bmm_all[i,:] = Bmm
    B0m_all[i,:] = B0m
    B00_all[i,:] = B00
    Bpm_all[i,:] = Bpm

# np.savetxt('./kernels/Bmm_all.dat',Bmm_all)
# np.savetxt('./kernels/B0m_all.dat',B0m_all)
# np.savetxt('./kernels/B00_all.dat',B00_all)
# np.savetxt('./kernels/Bpm_all.dat',Bpm_all)


np.savetxt('./cross_kernels/Bmm_all_s50.dat',Bmm_all)
np.savetxt('./cross_kernels/B0m_all_s50.dat',B0m_all)
np.savetxt('./cross_kernels/B00_all_s50.dat',B00_all)
np.savetxt('./cross_kernels/Bpm_all_s50.dat',Bpm_all)


plt.pcolormesh(Bmm_all)


####################################################

# l0_start = 4
# l0_end = 50
# n_max = 21  #modes from n=0 to n=21 have l=50.

# #because s=3, we have till \delta l = 3.
# Bmm_p1 = np.zeros((7,l0_end-l0_start,n_max+1,npts))
# Bmm_m1 = np.zeros((7,l0_end-l0_start,n_max+1,npts))
# Bmm_p2 = np.zeros((7,l0_end-l0_start,n_max+1,npts))
# Bmm_m2 = np.zeros((7,l0_end-l0_start,n_max+1,npts))
# Bmm_p3 = np.zeros((7,l0_end-l0_start,n_max+1,npts))
# Bmm_m3 = np.zeros((7,l0_end-l0_start,n_max+1,npts))

# B0m_p1 = np.zeros((7,l0_end-l0_start,n_max+1,npts))
# B0m_m1 = np.zeros((7,l0_end-l0_start,n_max+1,npts))
# B0m_p2 = np.zeros((7,l0_end-l0_start,n_max+1,npts))
# B0m_m2 = np.zeros((7,l0_end-l0_start,n_max+1,npts))
# B0m_p3 = np.zeros((7,l0_end-l0_start,n_max+1,npts))
# B0m_m3 = np.zeros((7,l0_end-l0_start,n_max+1,npts))

# B00_p1 = np.zeros((7,l0_end-l0_start,n_max+1,npts))
# B00_m1 = np.zeros((7,l0_end-l0_start,n_max+1,npts))
# B00_p2 = np.zeros((7,l0_end-l0_start,n_max+1,npts))
# B00_m2 = np.zeros((7,l0_end-l0_start,n_max+1,npts))
# B00_p3 = np.zeros((7,l0_end-l0_start,n_max+1,npts))
# B00_m3 = np.zeros((7,l0_end-l0_start,n_max+1,npts))

# Bpm_p1 = np.zeros((7,l0_end-l0_start,n_max+1,npts))
# Bpm_m1 = np.zeros((7,l0_end-l0_start,n_max+1,npts))
# Bpm_p2 = np.zeros((7,l0_end-l0_start,n_max+1,npts))
# Bpm_m2 = np.zeros((7,l0_end-l0_start,n_max+1,npts))
# Bpm_p3 = np.zeros((7,l0_end-l0_start,n_max+1,npts))
# Bpm_m3 = np.zeros((7,l0_end-l0_start,n_max+1,npts))

# #the first index denotes l0-l value. can be +/-1, +/-2, +/-3.
# # -1,-2,-3 will be stored in 6,5,4 indices.
# # 0th index will be blank

# nl_list = np.loadtxt('nl.dat')
# nl_arr = nl_list[nl_list[:,0]==0]
# nl_arr = nl_arr[nl_arr[:,1]>=l0_start-1]

# for i in range(1,n_max+1):   
#     nl_arr_temp = nl_list[nl_list[:,0]==i]
#     nl_arr = np.append(nl_arr,nl_arr_temp[nl_arr_temp[:,1]>=l0_start-1],axis=0)

# nl_arr = nl_arr.astype(int)

# n0 = 1

# for i in range(l0_start,l0_end,1):
#     nl_arr2 = nl_arr[nl_arr[:,1]==i+1]
#     l0 = i
#     for j in range(len(nl_arr2)):
#         n = nl_arr2[j,0]
#         l = nl_arr2[j,1]
#         Bmm, B0m,B00, Bpm,_,_ = np.array(gkerns.Hkernels(n0,l0,m,n,l,m,s,r)\
#                                 .ret_kerns_axis_symm())*plot_fac
#         Bmm_p1[l0-l,i-l0_start,j,:] = Bmm
#         B0m_p1[l0-l,i-l0_start,j,:] = B0m
#         B00_p1[l0-l,i-l0_start,j,:] = B00
#         Bpm_p1[l0-l,i-l0_start,j,:] = Bpm

#         print(i,j)

#     nl_arr2 = nl_arr[nl_arr[:,1]==i-1]
#     l0 = i
#     for j in range(len(nl_arr2)):
#         n = nl_arr2[j,0]
#         l = nl_arr2[j,1]
#         Bmm, B0m,B00, Bpm,_,_ = np.array(gkerns.Hkernels(n0,l0,m,n,l,m,s,r)\
#                                 .ret_kerns_axis_symm())*plot_fac
#         Bmm_m1[l0-l,i-l0_start,j,:] = Bmm
#         B0m_m1[l0-l,i-l0_start,j,:] = B0m
#         B00_m1[l0-l,i-l0_start,j,:] = B00
#         Bpm_m1[l0-l,i-l0_start,j,:] = Bpm

#     nl_arr2 = nl_arr[nl_arr[:,1]==i+2]
#     l0 = i
#     for j in range(len(nl_arr2)):
#         n = nl_arr2[j,0]
#         l = nl_arr2[j,1]
#         Bmm, B0m,B00, Bpm,_,_ = np.array(gkerns.Hkernels(n0,l0,m,n,l,m,s,r)\
#                                 .ret_kerns_axis_symm())*plot_fac
#         Bmm_p2[l0-l,i-l0_start,j,:] = Bmm
#         B0m_p2[l0-l,i-l0_start,j,:] = B0m
#         B00_p2[l0-l,i-l0_start,j,:] = B00
#         Bpm_p2[l0-l,i-l0_start,j,:] = Bpm

#         print(i,j)

#     nl_arr2 = nl_arr[nl_arr[:,1]==i-2]
#     l0 = i
#     for j in range(len(nl_arr2)):
#         n = nl_arr2[j,0]
#         l = nl_arr2[j,1]
#         Bmm, B0m,B00, Bpm,_,_ = np.array(gkerns.Hkernels(n0,l0,m,n,l,m,s,r)\
#                                 .ret_kerns_axis_symm())*plot_fac
#         Bmm_m2[l0-l,i-l0_start,j,:] = Bmm
#         B0m_m2[l0-l,i-l0_start,j,:] = B0m
#         B00_m2[l0-l,i-l0_start,j,:] = B00
#         Bpm_m2[l0-l,i-l0_start,j,:] = Bpm

#         print(i,j)

#     nl_arr2 = nl_arr[nl_arr[:,1]==i+3]
#     l0 = i
#     for j in range(len(nl_arr2)):
#         n = nl_arr2[j,0]
#         l = nl_arr2[j,1]
#         Bmm, B0m,B00, Bpm,_,_ = np.array(gkerns.Hkernels(n0,l0,m,n,l,m,s,r)\
#                                 .ret_kerns_axis_symm())*plot_fac
#         Bmm_p3[l0-l,i-l0_start,j,:] = Bmm
#         B0m_p3[l0-l,i-l0_start,j,:] = B0m
#         B00_p3[l0-l,i-l0_start,j,:] = B00
#         Bpm_p3[l0-l,i-l0_start,j,:] = Bpm

#         print(i,j)

#     nl_arr2 = nl_arr[nl_arr[:,1]==i-3]
#     l0 = i
#     for j in range(len(nl_arr2)):
#         n = nl_arr2[j,0]
#         l = nl_arr2[j,1]
#         Bmm, B0m,B00, Bpm,_,_ = np.array(gkerns.Hkernels(n0,l0,m,n,l,m,s,r)\
#                                 .ret_kerns_axis_symm())*plot_fac
#         Bmm_m3[l0-l,i-l0_start,j,:] = Bmm
#         B0m_m3[l0-l,i-l0_start,j,:] = B0m
#         B00_m3[l0-l,i-l0_start,j,:] = B00
#         Bpm_m3[l0-l,i-l0_start,j,:] = Bpm

#         print(i,j)

# B_m1 = np.zeros((4,))


# np.savetxt('./cross_kernels/Bmm_p1.dat',Bmm_p1)
# np.savetxt('./cross_kernels/B0m_p1.dat',B0m_p1)
# np.savetxt('./cross_kernels/B00_p1.dat',B00_p1)
# np.savetxt('./cross_kernels/Bpm_p1.dat',Bpm_p1)

# np.savetxt('./cross_kernels/Bmm_m1.dat',Bmm_m1)
# np.savetxt('./cross_kernels/B0m_m1.dat',B0m_m1)
# np.savetxt('./cross_kernels/B00_m1.dat',B00_m1)
# np.savetxt('./cross_kernels/Bpm_m1.dat',Bpm_m1)

# np.savetxt('./cross_kernels/Bmm_p2.dat',Bmm_p2)
# np.savetxt('./cross_kernels/B0m_p2.dat',B0m_p2)
# np.savetxt('./cross_kernels/B00_p2.dat',B00_p2)
# np.savetxt('./cross_kernels/Bpm_p2.dat',Bpm_p2)

# np.savetxt('./cross_kernels/Bmm_m2.dat',Bmm_m2)
# np.savetxt('./cross_kernels/B0m_m2.dat',B0m_m2)
# np.savetxt('./cross_kernels/B00_m2.dat',B00_m2)
# np.savetxt('./cross_kernels/Bpm_m2.dat',Bpm_m2)

# np.savetxt('./cross_kernels/Bmm_p3.dat',Bmm_p3)
# np.savetxt('./cross_kernels/B0m_p3.dat',B0m_p3)
# np.savetxt('./cross_kernels/B00_p3.dat',B00_p3)
# np.savetxt('./cross_kernels/Bpm_p3.dat',Bpm_p3)

# np.savetxt('./cross_kernels/Bmm_m3.dat',Bmm_m3)
# np.savetxt('./cross_kernels/B0m_m3.dat',B0m_m3)
# np.savetxt('./cross_kernels/B00_m3.dat',B00_m3)
# np.savetxt('./cross_kernels/Bpm_m3.dat',Bpm_m3)
