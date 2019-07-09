import functions as fn
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import submatrix

OM = np.loadtxt('OM.dat')
r = np.loadtxt('r.dat')
n,l = 0,200
m = np.arange(-l,l+1)
omega_ref = np.loadtxt('muhz.dat')[fn.find_nl(n,l)] * 1e-6 /OM

a = submatrix.diffrot(n,n,l,l,r,omega_ref)
b = np.loadtxt('atharv_data/omegs200')

#a = np.sqrt(omega_ref**2 + a.astype('float64')) * 1e6 *OM
a = (omega_ref + a.astype('float64') / (2.*omega_ref)) * 1e6 *OM

plt.subplot(211)
plt.plot(m,a,label = 'dpt')
plt.plot(m,b,label = 'obs')
plt.legend()
plt.grid(True)

plt.subplot(212)
plt.plot(m,a-b,label = 'dpt - obs')
plt.legend()
plt.grid(True)

del_omega_a = a - omega_ref * OM * 1e6
del_omega_b = b - omega_ref * OM * 1e6
print fn.a_coeff(del_omega_a,l,5)
print fn.a_coeff(del_omega_b,l,5)

plt.show()
