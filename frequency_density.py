import numpy as np
import functions as fn
import matplotlib.pyplot as plt

omega = np.loadtxt('muhz.dat')
#omega_sorted = np.sort(omega)
#om_hist = np.histogram(omega_sorted, bins = 3000)

#plt.hist(omega, bins = 3000)
#plt.show()

for i in range(len(omega)):
    l1 = fn.find_mode(i)[1]
    for j in range(i+1,len(omega)):
        l2 = fn.find_mode(j)[1]
#        print i,j
        if(np.abs(omega[i] - omega[j]) < 0.5 and max(l1,l2)<10 and np.abs(l1-l2) < 6 and np.abs(l1-l2)%2 == 0):
            print fn.find_mode(i),fn.find_mode(j), np.abs(omega[i] - omega[j])



