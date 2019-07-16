import numpy as np
import functions as fn
import matplotlib.pyplot as plt
import sys

omega_list = np.loadtxt('muhz.dat')
nl_list = np.loadtxt('nl.dat')

om_sorted_ind = np.argsort(omega_list)
width = 200
eps = 10
ind_om = list(zip(om_sorted_ind, np.take(omega_list,om_sorted_ind)))

for i in range(len(omega_list)):
    mode_set = [fn.find_mode(ind_om[i][0])]
    omega0 = ind_om[i][1]
    l0 = fn.find_mode(ind_om[i][0])[1]
    #for j in range(max(0,i-width), min(len(omega_list), i+width+1)):
    for j in range(len(omega_list)):
        omega1 = ind_om[j][1]
        l1 = fn.find_mode(ind_om[j][0])[1]        
        if (np.abs(omega0 - omega1) < eps and i != j and np.abs(l0-l1) < width and (l0-l1)%2==0 and l0 <100 and l0 > 20):
            mode_set.append(fn.find_mode(ind_om[j][0]))
    if(len(mode_set)>5):    
        print(mode_set)
        omega_set = [omega_list[fn.find_nl(mode[0], mode[1])] for mode in mode_set]
        print(omega_set)
