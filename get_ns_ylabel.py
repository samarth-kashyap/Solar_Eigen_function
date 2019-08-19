import numpy as np
import matplotlib.pyplot as plt
from os import getcwd
import functions as fn
import get_kernels as gkerns
import sys

r = np.loadtxt('./r.dat')
npts = 300

#starting from r = 0.5 R_sol
start_ind = fn.nearest_index(r,0.8)
end_ind = fn.nearest_index(r,1.)  #ends at the surface
r = r[start_ind:end_ind+1]
OM = np.loadtxt('./OM.dat')

#self coupling
m = np.array([0])
m_ = np.array([0])
s = np.array([2])

nl_list = np.loadtxt('./nl.dat')
nl_arr = nl_list[nl_list[:,0]==0]
nl_arr = nl_arr[nl_arr[:,1]>s[0]]

plot_fac = OM**2 * 1e12 * (4.*np.pi/3) * 1e-10 #unit = muHz^2 G^(-2) V_sol^(-1)

#rearranging so that we can iterate for all l's over a fixed n's
for i in range(1,33):   #max n goes upto 32
    nl_arr_temp = nl_list[nl_list[:,0]==i]
    nl_arr = np.append(nl_arr,nl_arr_temp[nl_arr_temp[:,1]>s[0]],axis=0)

nl_arr = nl_arr.astype(int)

ytickloc = np.zeros((1,2))

start_n = 0
end_n = 31

cumulative_nl = 0

for i in range(start_n,end_n):
    nl_arr_temp2 = nl_arr[nl_arr[:,0]==i]
    tickloc = cumulative_nl 
    tickloc += (nl_arr_temp2[0,1] + nl_arr_temp2[-1,1])/2 - nl_arr_temp2[0,1]
    ytickloc = np.append(ytickloc,np.array([[i,int(tickloc)]]),axis=0)

    cumulative_nl += np.amax(nl_arr_temp2)

    print(cumulative_nl)

ytickloc = ytickloc[1:]

np.savetxt('ytickinfo.dat',ytickloc)