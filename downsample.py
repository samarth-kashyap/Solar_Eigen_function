import numpy as np
import matplotlib.pyplot as plt
import functions as fn
import scipy.interpolate as interpolate
import sys

r = np.loadtxt('r_full.dat')
rho = np.loadtxt('rho_full.dat')
nl_list = np.loadtxt('nl.dat')


n_pts = 105
grid_ind = np.arange(0,len(r),1+len(r)/n_pts)
#print np.arange(len(r))
#sys.exit()
r_new = interpolate.interp1d(np.arange(len(r)),r,kind= 'cubic')(grid_ind)
rho_new = interpolate.interp1d(np.arange(len(r)),rho,kind= 'cubic')(grid_ind)

#r_new = np.zeros(199)
#r_new[:98] = r[0:4900:50]
#r_new[98:] = r[4900:len(r):24]

for nl in range(len(nl_list)):
    n,l = nl_list[nl]
    Ui, Vi = fn.load_eig(n,l,'eig_files_full')
    
    Ui_interp = interpolate.interp1d(r,Ui,kind='cubic')
    Vi_interp = interpolate.interp1d(r,Vi,kind='cubic')
    
    U_new = Ui_interp(r_new)
    V_new = Vi_interp(r_new)
    Uname = 'U'+str(nl)
    Vname = 'V'+str(nl)    
    np.savetxt('eig_files/' + Uname+'.dat', U_new)
    np.savetxt('eig_files/' + Vname+'.dat', V_new)
    print fn.find_mode(nl)
    
np.savetxt('r.dat',r_new)    
np.savetxt('rho.dat',rho_new)    
#plt.plot(r_new,U_new,'.')
plt.show()
