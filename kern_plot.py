import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

nl_list = np.loadtxt('nl.dat')

n,l = 4,2
n_,l_ = 3,2
nl = None
nl_ = None

for i in range(len(nl_list)):	
	if (np.array_equal(nl_list[i],np.array([n,l]))):	nl = i
	if (np.array_equal(nl_list[i],np.array([n_,l_]))):	nl_ = i
if(nl == None or nl_ == None):
	print("Mode not found. Exiting.")
	exit()

U, V = np.loadtxt('eigU.dat')[nl], np.loadtxt('eigV.dat')[nl]
U_,V_ = np.loadtxt('eigU.dat')[nl_], np.loadtxt('eigV.dat')[nl_]
r = np.loadtxt('r.dat')
rho = np.loadtxt('rho.dat')

print integrate.simps(rho*(U_*U+l*(l+1.)*V_*V),r, even= 'avg')

#plt.plot(r,U*U,'r-')
plt.plot(r,V*V,'b-')
plt.show()
