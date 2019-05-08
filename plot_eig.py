import numpy as np
import matplotlib.pyplot as plt
import os 

nl_list = np.loadtxt('nl.dat')
U_list = np.loadtxt('eigU.dat')
V_list = np.loadtxt('eigV.dat')
r = np.loadtxt('r.dat')
rho = np.loadtxt('rho.dat')
nl = 0
for nl in range(100):
	n,l = np.int64(nl_list[nl])
	U, V = U_list[nl], V_list[nl]
	plt.figure()
	plt.plot(r,U,'r-')
	plt.plot(r,V,'b-')
	name = str(n)+'_'+str(l)
	plt.savefig(os.getcwd()+'/plots/'+name+'.png')
exit()
