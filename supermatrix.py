import numpy as np
import matplotlib.pyplot as plt
import submatrix
import functions as fn
import sys
import timing
import matplotlib.gridspec as gridspec

clock1 = timing.stopclock()
tstamp = clock1.lap
#all quantities in cgs
#M_sol = 1.989e33 g
#R_sol = 6.956e10 cm
#B_0 = 10e5 G
#OM = np.sqrt(4*np.pi*R_sol*B_0**2/M_sol)

OM = np.loadtxt('OM.dat') #importing normalising frequency value from file (in Hz (cgs))

field_type = 'dipolar'
r = np.loadtxt('r.dat')
r_start, r_end = 0.,1.
start_ind, end_ind = [fn.nearest_index(r, pt) for pt in (r_start, r_end)]
r = r[start_ind:end_ind]

nl_list = np.array([[0,198],[0,200],[0,202]])
#nl_list = np.array([[0,2],[0,3]])
omega_list = np.loadtxt('muhz.dat') * 1e-6 / OM #normlaised frequency list
omega_nl = np.array([omega_list[fn.find_nl(mode[0], mode[1])] for mode in nl_list])
omega_ref0 = np.mean(omega_nl)

#print omega_nl
#sys.exit()

total_m = len(nl_list) + 2*np.sum(nl_list, axis = 0)[1]

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
        temp_matrix = submatrix.diffrot(n_,n,l_,l,r,omega_ref0)
        #print(temp_matrix)
        temp_matrix = np.diag(temp_matrix)
#        temp_matrix = np.diag(submatrix.submatrix_diagonal_diffrot(n_,l_,r,omega_nl[i]))
        del_i, del_j = l_-np.amin([l_,l]), l-np.amin([l_,l])
        temp_matrix = np.pad(temp_matrix,((del_i,del_i),(del_j,del_j)),'constant',\
                      constant_values = 0)                
        Z[mi_beg:mi_end,mj_beg:mj_end] = temp_matrix
        
        print('((%i,%i),(%i,%i))'%(n_,l_,n,l))
        tstamp('done in:')
        mj_beg += 2*l+1
                
    Z_diag[mi_beg:mi_end,mi_beg:mi_end] *= -(omega_ref0**2 - omega_nl[i]**2)
    Z_dpt[mi_beg:mi_end,mi_beg:mi_end] = Z[mi_beg:mi_end,mi_beg:mi_end]\
                                 +  np.identity(mi_end-mi_beg) *omega_nl[i]**2
    mi_beg += 2*l_+1

eig_vals_dpt = np.diag(Z_dpt)
#inserting the diagonal component of the supermatrix
Z += Z_diag   
eig_vals_qdpt,eig_vts_qdpt = np.linalg.eig(Z)
m_ind_arr = np.zeros(np.shape(eig_vals_qdpt),dtype=int)

for i in range(len(eig_vals_qdpt)):
    m_ind_arr[i] = np.argmax(np.abs(eig_vts_qdpt[i]))

eig_vals_qdpt_arranged = eig_vals_qdpt[m_ind_arr]

#######################
#Extracting frequencies

f_dpt = np.sqrt(eig_vals_dpt) * OM *1e6
f_qdpt = np.sqrt(omega_ref0**2 + eig_vals_qdpt_arranged) * OM *1e6

#######################
#Creating list of all original m's wrt ordering in nl
#Could be useful for labeling the x-axis

plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(2,1,1)

plt.plot(f_dpt,label='Degenerate')
plt.plot(f_qdpt,label='Quasi-Degenerate')
plt.legend()

f_dpt_min = np.amin(f_dpt)
f_dpt_max = np.amax(f_dpt)
freq_arr = np.arange(f_dpt_min,f_dpt_max,(f_dpt_max-f_dpt_min)/100)

l_local = 0
title_str = ''
for i in range(len(nl_list)-1):
    l_local += (2*nl_list[i,1]+1)
    plt.plot(l_local*np.ones(len(freq_arr)),freq_arr,'--k',alpha = 0.3)
    title_str = title_str + 'n,l = ' + str(nl_list[i,0]) + ',' + str(nl_list[i,1]) + ';'

title_str = title_str + 'n,l = ' + str(nl_list[-1,0]) + ',' + str(nl_list[-1,1]) + ';'

plt.title(title_str)
plt.ylabel('Frequency in $\mu$Hz',fontsize=14)
plt.xlabel('Cumulative m',fontsize=12)

plt.subplot(2,1,2)

erf_dpt_min = np.amin(f_dpt-f_qdpt)
erf_dpt_max = np.amax(f_dpt-f_qdpt)
erfreq_arr = np.arange(erf_dpt_min,erf_dpt_max,(erf_dpt_max-erf_dpt_min)/100)

l_local = 0
for i in range(len(nl_list)-1):
    l_local += (2*nl_list[i,1]+1)
    plt.plot(l_local*np.ones(len(erfreq_arr)),erfreq_arr,'--k',alpha = 0.3)

plt.plot(f_dpt-f_qdpt)
plt.ylabel('$f_D - f_{QD}$ in $\mu$Hz',fontsize=14)
plt.xlabel('Cumulative m',fontsize=12)
plt.show()

sys.exit()

#MAGNETIC PERTUBATION

mi_beg = 0
for i in range(len(nl_list)):
    mj_beg = 0
    for j in range(len(nl_list)):
        n_,l_ = nl_list[i]
        n,l = nl_list[j]
       
        mi_end = mi_beg + (2*l_+1)
        mj_end = mj_beg + (2*l+1)
       
        temp_matrix = submatrix.lorentz_diagonal(n_,n,l_,l,r,field_type)
       
        temp_matrix = np.diag(temp_matrix)
        del_i, del_j = l_-np.amin([l_,l]), l-np.amin([l_,l])
        temp_matrix = np.pad(temp_matrix,((del_i,del_i),(del_j,del_j)),'constant',\
                      constant_values = 0)         
        Z[mi_beg:mi_end,mj_beg:mj_end] = temp_matrix                             
       
        print('((%i,%i),(%i,%i))'%(n_,l_,n,l))
       
        mj_beg += 2*l+1
       
    tstamp('omega_nlm starts')
    domega_nlm_sq = np.diag(submatrix.diffrot(n_,n_,l_,l_,r,omega_ref))
    domega_nlm_sq = domega_nlm_sq.astype('complex128')
    omega_nlm_sq = domega_nlm_sq + np.identity(mi_end-mi_beg) *omega_nl[i]**2
    tstamp('omega_nlm ends')
    
    omega_nlm_sq = omega_nlm_sq.astype('complex128')
        
    omega_ref = np.identity(mi_end-mi_beg)*omega_ref0
    Z_diag[mi_beg:mi_end,mi_beg:mi_end] += -(omega_ref**2 - omega_nlm_sq**2)
    Z_dpt[mi_beg:mi_end,mi_beg:mi_end] = Z[mi_beg:mi_end,mi_beg:mi_end]\
                                 +  np.identity(mi_end-mi_beg) *omega_nl[i]**2
    mi_beg += 2*l_+1



#######################
#Extracting frequencies

f_dpt = np.sqrt(eig_vals_dpt) * OM *1e6
f_qdpt = np.sqrt(omega_ref**2 + eig_vals_qdpt) * OM *1e6

#######################

plt.pcolormesh(np.log(np.abs(Z)))
plt.gca().invert_yaxis()
plt.colorbar()
plt.show('Block')

#ac = np.loadtxt('atharv_data/omegs200')
ob = np.sort(f_dpt)
#plt.plot(ac,'-' )
plt.plot(ob,'.',label='Degenerate')
plt.plot(np.sort(f_qdpt),'-',label='Quasi-degenerate')
print(np.max(ob)-np.min(ob))
plt.legend()
plt.show()
