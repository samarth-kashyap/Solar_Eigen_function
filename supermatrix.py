import numpy as np
import matplotlib.pyplot as plt
import submatrix
import functions as fn
import sys

#all quantities in cgs
#M_sol = 1.989e33 #g
#R_sol = 6.956e10 #cm
#B_0 = 10e5 #G
#OM = np.sqrt(4*np.pi*R_sol*B_0**2/M_sol)

OM = np.loadtxt('OM.dat') #importing normalising frequency value from file (in Hz (cgs))

field_type = 'dipolar'

n = 5
l=50

r = np.loadtxt('r.dat')
r_start, r_end = 0.99, 0.991
start_ind, end_ind = [fn.nearest_index(r, pt) for pt in (r_start, r_end)]
r = r[start_ind:end_ind]

nl_list = np.array([[4,5], [3,9]])
omega_list = np.loadtxt('muhz.dat') * 1e-6 / OM #normlaised frequency list
omega_nl = [omega_list[fn.find_nl(mode[0], mode[1])] for mode in nl_list]
omega_ref = np.mean(omega_nl)

#print omega_nl
#sys.exit()

total_m = len(nl_list) + 2*np.sum(nl_list, axis = 0)[1]

Z = np.empty((total_m, total_m))
Z_diag = np.identity(total_m)

mi_beg = 0
for i in range(len(nl_list)):
    mj_beg = 0
    for j in range(len(nl_list)): 
        n_,l_ = nl_list[i]
        n,l = nl_list[j]
        
        mi_end = mi_beg + (2*l_+1)
        mj_end = mj_beg + (2*l+1)
        
#        temp_matrix = submatrix.submatrix(n_,n,l_,l,r,45.,field_type)
        temp_matrix = submatrix.submatrix_diffrot(n_,n,l_,l,r)
        
        Z[mi_beg:mi_end,mj_beg:mj_end] = temp_matrix
        
        print('n,l:%i,%i'%(n,l))
        print('n_,l_:%i,%i'%(n_,l_))
        
        mj_beg += 2*l+1
        
    Z_diag[mi_beg:mi_end,mi_beg:mi_end] *= -(omega_ref**2 - omega_nl[i]**2)
    
    mi_beg += 2*l_+1
     
####OMLY FOR TESTING
Z *= 6e2
####ONLY FOR TESTING


#inserting the diagonal component of the supermatrix
Z += Z_diag   

       
#Z *= OM**2

#mm_,mm = np.meshgrid(m_,m,indexing='ij')
plt.pcolormesh(Z)
plt.gca().invert_yaxis()
plt.colorbar()
plt.show('Block')
plt.close()

eig_vals,_ = np.linalg.eig(Z)
####OMLY FOR TESTING
eig_vals /= 2.* omega_ref  
eig_vals *= OM *1e6
####ONLY FOR TESTING
    
plt.plot(np.sort(eig_vals),'-')
plt.show()
