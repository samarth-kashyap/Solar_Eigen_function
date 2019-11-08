import numpy as np
import matplotlib.pyplot as plt
import submatrix
import functions as fn
import sys
import timing
import matplotlib.gridspec as gridspec
import os.path
plt.ion()

clock1 = timing.stopclock()
tstamp = clock1.lap
#all quantities in cgs
#M_sol = 1.989e33 g
#R_sol = 6.956e10 cm
#B_0 = 10e5 G
#OM = np.sqrt(4*np.pi*R_sol*B_0**2/M_sol)
#rho_0 = M_sol/(4pi R_sol^3/3) = 1.41 ~ 1g/cc (for kernel calculation)

OM = np.loadtxt('OM.dat') #importing normalising frequency value from file (in Hz (cgs))

field_type = 'mixed'
r = np.loadtxt('r.dat')
r_start, r_end = 0.,1.
start_ind, end_ind = [fn.nearest_index(r, pt) for pt in (r_start, r_end)]
r = r[start_ind:end_ind]

nl_all = np.loadtxt('nl.dat')
omega_list = np.loadtxt('muhz.dat') * 1e-6 / OM #normlaised frequency list

#the frequency window around each mode
f_window = 100.0 #in muHz

s = np.array([0,1,2])
t = np.arange(-np.amax(s),np.amax(s)+1)
# t = np.array([0])
# n0 = 6  #choosing the branch

# l_max = int(np.amax(l_n0) - np.amax(s))
# l_max = int(np.amax(l_n0))
l_max = 30

#max s values for DR and magnetic field to speed up supermatrix computation
s_max_H = np.max(s)     #axisymmteric magnetic field. B has only s = 1. 

#if we want to use smoothened kernels
smoothen = False

#4 components of munu as we ar clubbing together (-- with ++) and (0- with 0+)
qdpt_contrib_rel = np.zeros((4,300))
# qdpt_contrib = np.zeros((4,300))

for n0 in range(0,20):
    nl_list_n0 = nl_all[nl_all[:,0]==n0]
    l_n0 = nl_list_n0[:,1]

    l_min = int(np.amin(l_n0) + np.amax(s))
    #running over all the central modes whose frequency shifts we want to find
    for l0 in range(l_min,l_max+1):
        # nl_list = np.zeros((2*np.amax(s)+1,2))  #modes to be considered for a certain central mode
        # nl_list[:,0] = n0   #branch to which it should belong to
        # nl_list[:,1] = np.arange(l0-s_max_H,l0+s_max_H+1) #the +/- s_H_max around l0
        # nl_list = nl_list.astype('int64')   #making the modes integers for future convenience

        omega_nl0 = omega_list[fn.find_nl(n0, l0)]
        # omega_ref0 = np.mean(omega_nl)  #finding the central frequency for doing QDPT
        omega_ref0 = omega_nl0  #setting the delta omegas of the central mode to be around zero

        nl_list = fn.nearest_freq_modes(l0,s_max_H,omega_nl0,f_window)
        nl_list = nl_list.astype('int64')   #making the modes integers for future convenience
        
        omega_nl = np.array([omega_list[fn.find_nl(mode[0], mode[1])] for mode in nl_list]) #important to have nl_list as integer type
        nearest_omega_jump = np.amin(np.sort(np.abs(omega_nl-omega_nl0))[1])

        total_m = len(nl_list) + 2*np.sum(nl_list, axis = 0)[1] #total length of supermatrix

        #MAGNETIC PERTUBATION
        Z_large = np.zeros((4,total_m, total_m),dtype='complex128')
        Z_diag = np.identity(total_m,dtype='complex128')    #to store the omega_nl**2 - omega_ref0**2
        Z_dpt = np.zeros((4,2*l0+1, 2*l0+1),dtype='complex128')  #for self-coupling approximation
        mi_beg = 0

        for i in range(len(nl_list)):
            mj_beg = 0
            n_,l_ = nl_list[i]
            mi_end = mi_beg + (2*l_+1)
            for j in range(len(nl_list)):
                n,l = nl_list[j]
                mj_end = mj_beg + (2*l+1)
                print('l=%i, ((%i,%i),(%i,%i))'%(l0,n_,l_,n,l))

                #implementing selection rule: delta_l <= s_max
                if(np.abs(l-l_)>s_max_H): 
                    mj_beg += 2*l+1
                    print('Skipping')
                    continue
                
                if(os.path.exists('Simulation_submatrices/%i_%i_%i_%i.npy'%(n_,l_,n,l))):
                    print('Computed mode exists')
                    Z_large[:,mi_beg:mi_end,mj_beg:mj_end] = np.load('Simulation_submatrices/%i_%i_%i_%i.npy'%(n_,l_,n,l))
                else:
                    Z_large[:,mi_beg:mi_end,mj_beg:mj_end] = submatrix.lorentz_all_st_equalB(n_,n,l_,l,r,s,t)  
                    np.save('Simulation_submatrices/%i_%i_%i_%i.npy'%(n_,l_,n,l),Z_large[:,mi_beg:mi_end,mj_beg:mj_end])                       
            
                mj_beg += 2*l+1

                
            Z_diag[mi_beg:mi_end,mi_beg:mi_end] *= (omega_nl[i]**2 - omega_ref0**2)

            mi_beg += 2*l_+1
    

        Herm_res = Z_large - np.conj(np.transpose(Z_large,(0,2,1)))
        if(np.amax(np.abs(Herm_res.real)) > 1e-12 or np.amax(np.abs(Herm_res.imag)) > 1e-12): print('Supermatrix is not Hermitian')

    # Zmunu_dpt = np.load('Simulation_submatrices/%i_%i_%i_%i.npy'%(n0,l0,n0,l0)) #File should exist at this point 
    # #finding the relative cross-coupling contributions for a particular munu and s
    # for munu in range(4):
    #     title ='Clean'  #Initializing the case as clean. Assuming that the modes are well spaced after splitting
    #     Z = Z_large[munu,:,:]   #Isolating the component of H whose splitting we want to find.

    #     #Need to do this only for the central mode for DPT
    #     Z_dpt = Zmunu_dpt[munu,:,:]  

    #     eig_vals_dpt,eig_vts_dpt = np.linalg.eig(Z_dpt)     #Finding the frequency shifts only due to an isolated multiplet

    #     #inserting the diagonal component of the supermatrix
    #     Z += Z_diag   #Adding the diagonal for QDPT
    #     eig_vals_qdpt,eig_vts_qdpt = np.linalg.eigh(Z)

    #     #Initializing variables for analyzing mode spacing after splitting.
    #     evals_qdpt_diff_abssorted = np.diff(np.sort(np.abs(eig_vals_qdpt)))
    #     cent_mode_SD = np.sqrt(np.var(evals_qdpt_diff_abssorted[:2*l0]))
    #     freq_jump = evals_qdpt_diff_abssorted[2*l0]/(2*omega_ref0)

    #     #Checking if we should already discard some modes which come too close or overlap.
    #     if(freq_jump <= nearest_omega_jump/4 or cent_mode_SD/freq_jump > 0.3):
    #         qdpt_contrib_rel[munu,l0] = 100.0
    #         title ='Unclean'    #Marking them unclean to label the plots
            

    #     #######################
    #     #Extracting frequencies

    #     l_local = 0
    #     omega_nl_arr = np.zeros(total_m)
    #     for i in range(len(nl_list)):
    #         omega_nl_arr[l_local:l_local + 2*nl_list[i,1]+1] = \
    #                     np.ones(2*nl_list[i,1]+1)*omega_nl[i]
    #         l_local += 2*nl_list[i,1]+1

    #     f_dpt = (omega_nl0 + eig_vals_dpt/(2*omega_nl0)) * OM *1e6
    #     f_qdpt = np.sqrt(omega_ref0**2 + eig_vals_qdpt) * OM *1e6

    #     plt.subplot(211)
    #     plt.plot(np.sort(np.abs(eig_vals_qdpt)),'.')
    #     plt.title('%s'%title)

    #     plt.subplot(212)
    #     plt.plot(evals_qdpt_diff_abssorted,'.')

    #     plt.savefig('./Cross_coupling_goodness/l_%i_munu_%i'%(l0,munu))

    #     plt.close()

            