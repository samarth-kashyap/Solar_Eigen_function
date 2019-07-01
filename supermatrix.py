import numpy as np
import matplotlib.pyplot as plt
import submatrix
import functions as fn
import sys
import timing
clock1 = timing.stopclock()
tstamp = clock1.lap
#all quantities in cgs
#M_sol = 1.989e33 g
#R_sol = 6.956e10 cm
#B_0 = 10e5 G
#OM = np.sqrt(4*np.pi*R_sol*B_0**2/M_sol)

OM = np.loadtxt('OM.dat') #importing normalising frequency value from file (in Hz (cgs))

field_type = 'dipolar'

n = 5
l=5

r = np.loadtxt('r.dat')
r_start, r_end = 0.,1.
start_ind, end_ind = [fn.nearest_index(r, pt) for pt in (r_start, r_end)]
r = r[start_ind:end_ind]

nl_list = np.array([[0,200],[0,198],[0,202]])
omega_list = np.loadtxt('muhz.dat') * 1e-6 / OM #normlaised frequency list
omega_nl = np.array([omega_list[fn.find_nl(mode[0], mode[1])] for mode in nl_list])
omega_ref = np.sqrt(np.mean(omega_nl**2))

#print omega_nl
#sys.exit()

total_m = len(nl_list) + 2*np.sum(nl_list, axis = 0)[1]

#Z = np.empty((total_m, total_m))
#Z_diag = np.identity(total_m)
#Z_dpt = np.zeros((total_m, total_m))
#mi_beg = 0
#for i in range(len(nl_list)):
#    mj_beg = 0
#    n_,l_ = nl_list[i]    
#    mi_end = mi_beg + (2*l_+1)
#    '''for j in range(len(nl_list)): 
#        n_,l_ = nl_list[i]
#        n,l = nl_list[j]
#        
#        mi_end = mi_beg + (2*l_+1)
#        mj_end = mj_beg + (2*l+1)
#        
##        temp_matrix = submatrix.submatrix(n_,n,l_,l,r,45.,field_type)
##        temp_matrix = submatrix.submatrix_diffrot(n_,n,l_,l,r,omega_ref)
#        temp_matrix = np.diag(submatrix.submatrix_diagonal_diffrot(n_,l_,r,omega_nl[i]))
#                
#        Z[mi_beg:mi_end,mj_beg:mj_end] = temp_matrix
#        
#        print('n,l:%i,%i'%(n,l))
#        print('n_,l_:%i,%i'%(n_,l_))
#        
#        mj_beg += 2*l+1'''
#        

#    Z[mi_beg:mi_end,mi_beg:mi_end] = np.diag(submatrix.submatrix_diagonal_diffrot(n_,l_,r,omega_nl[i]))
##    Z_diag[mi_beg:mi_end,mi_beg:mi_end] *= -(omega_ref**2 - omega_nl[i]**2)
#    Z_dpt[mi_beg:mi_end,mi_beg:mi_end] = Z[mi_beg:mi_end,mi_beg:mi_end]\
#                                 +  np.identity(mi_end-mi_beg) *omega_nl[i]**2
#    mi_beg += 2*l_+1


Z = np.empty((total_m, total_m))
Z_diag = np.identity(total_m)
Z_dpt = np.zeros((total_m, total_m))
mi_beg = 0
for i in range(len(nl_list)):
    mj_beg = 0
    for j in range(len(nl_list)): 
        tstamp()
        n_,l_ = nl_list[i]
        n,l = nl_list[j]
        
        mi_end = mi_beg + (2*l_+1)
        mj_end = mj_beg + (2*l+1)
        
#        temp_matrix = submatrix.submatrix(n_,n,l_,l,r,45.,field_type)
        temp_matrix = submatrix.submatrix_diffrot(n_,n,l_,l,r,omega_ref)
        temp_matrix = np.diag(temp_matrix)
#        temp_matrix = np.diag(submatrix.submatrix_diagonal_diffrot(n_,l_,r,omega_nl[i]))
        del_i, del_j = l_-np.amin([l_,l]), l-np.amin([l_,l])
        temp_matrix = np.pad(temp_matrix,((del_i,del_i),(del_j,del_j)),'constant',\
                      constant_values = 0)                
        Z[mi_beg:mi_end,mj_beg:mj_end] = temp_matrix
        
        print('((%i,%i),(%i,%i))'%(n_,l_,n,l))
        tstamp('done in:')
        mj_beg += 2*l+1
        
        
    Z_diag[mi_beg:mi_end,mi_beg:mi_end] *= -(omega_ref**2 - omega_nl[i]**2)
    Z_dpt[mi_beg:mi_end,mi_beg:mi_end] = Z[mi_beg:mi_end,mi_beg:mi_end]\
                                 +  np.identity(mi_end-mi_beg) *omega_nl[i]**2
    mi_beg += 2*l_+1
     
#####OMLY FOR TESTING
#Z *= 6e2
#####ONLY FOR TESTING

eig_vals_dpt = np.sort(np.diag(Z_dpt))
#inserting the diagonal component of the supermatrix
Z += Z_diag   
eig_vals_qdpt,_ = np.linalg.eig(Z)


#######################
#Extracting frequencies

f_dpt = np.sqrt(eig_vals_dpt) * OM *1e6

f_qdpt = (omega_ref + eig_vals_qdpt/(2.*omega_ref)) * OM *1e6

#######################

plt.pcolormesh(np.log(np.abs(Z)))
plt.gca().invert_yaxis()
plt.colorbar()
plt.show('Block')

#####OMLY FOR DISPLAYING
#eig_vals /= (2.* omega_ref)  
#eig_vals *= OM *1e6
#####ONLY FOR DISPLAYING

#ac = np.loadtxt('atharv_data/omegs200')
ob = np.sort(f_dpt)
#plt.plot(ac,'-' )
plt.plot(ob,'-',label='Degenerate')
plt.plot(np.sort(f_qdpt),'-',label='Quasi-degenerate')
print np.max(ob)-np.min(ob)
plt.legend()
plt.show()
