import numpy as np
import matplotlib.pyplot as plt
import submatrix
import functions as fn
import sys

#all quantities in cgs
M_sol = 1.989e33 #g
R_sol = 6.956e10 #cm
B_0 = 10e5 #G
OM = np.sqrt(4*np.pi*R_sol*B_0**2/M_sol)
field_type = 'dipolar'

n = 5
l=30

r = np.loadtxt('r.dat')
r_start, r_end = 0.6, 0.601
start_ind, end_ind = [fn.nearest_index(r, pt) for pt in (r_start, r_end)]
r = r[start_ind:end_ind]

nl_list = np.array([[n,l-1],[n,l],[n,l+1],[n,l+2]])
total_m = len(nl_list) + 2*np.sum(nl_list, axis = 0)[1]

Z = np.empty((total_m, total_m))

mi_beg = 0
for i in range(len(nl_list)):
    mj_beg = 0
    for j in range(len(nl_list)): 
        n_,l_ = nl_list[i]
        n,l = nl_list[j]
        
        mi_end = mi_beg + (2*l_+1)
        mj_end = mj_beg + (2*l+1)
        
        temp_matrix = submatrix.submatrix(n_,n,l_,l,r,45.,field_type)
        
        Z[mi_beg:mi_end,mj_beg:mj_end] = temp_matrix
        print('n,l:%i,%i'%(n,l))
        print('n_,l_:%i,%i'%(n_,l_))
        #print(np.shape(Z[mi_beg:mi_end,mj_beg:mj_end]))
        #print(np.shape(temp_matrix))
        
        mj_beg += 2*l+1
    
    mi_beg += 2*l_+1
     
    
        
Z *= OM**2 / 1e-3        

#mm_,mm = np.meshgrid(m_,m,indexing='ij')
plt.pcolormesh(Z)
plt.gca().invert_yaxis()
plt.colorbar()
plt.show('Block')

    
