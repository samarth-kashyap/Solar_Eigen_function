import numpy as np
import functions as fn
import matplotlib.pyplot as plt
import sys

omega_list = np.loadtxt('muhz.dat')
nl_list = np.loadtxt('nl.dat')

om_sorted_ind = np.argsort(omega_list)
width = 200
eps = 25.
ind_om = list(zip(om_sorted_ind, np.take(omega_list,om_sorted_ind)))

for i in range(len(omega_list)):
    mode_central = fn.find_mode(ind_om[i][0])
    mode_set = [[mode_central[0],mode_central[1]]]
    omega0 = ind_om[i][1]
    l0 = fn.find_mode(ind_om[i][0])[1]
    for j in range(max(0,i-width), min(len(omega_list), i+width+1)):
        omega1 = ind_om[j][1]
        l1 = fn.find_mode(ind_om[j][0])[1]        
        if (np.abs(omega0 - omega1) < eps and i != j and np.abs(l0-l1) < 6 and (l0-l1)%2==0 and l0 <100 and l0 > 20):
            new_mode = fn.find_mode(ind_om[j][0])
            new_mode_arr = [new_mode[0],new_mode[1]]
            mode_set.append(new_mode_arr)
    if(len(mode_set)>4):    
        print(mode_set)
        omega_set = [omega_list[fn.find_nl(mode[0], mode[1])] for mode in mode_set]
        print(omega_set)
