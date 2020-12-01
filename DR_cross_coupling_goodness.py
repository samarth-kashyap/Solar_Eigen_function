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

####################################################################
#all quantities in cgs
#M_sol = 1.989e33 g
#R_sol = 6.956e10 cm
#B_0 = 10e5 G
#OM = np.sqrt(4*np.pi*R_sol*B_0**2/M_sol)
#rho_0 = M_sol/(4pi R_sol^3/3) = 1.41 ~ 1g/cc (for kernel calculation)
#####################################################################

OM = np.loadtxt('OM.dat') #importing normalising frequency value from file (in Hz (cgs))

field_type = 'mixed'
r = np.loadtxt('r.dat')
r_start, r_end = 0.,1.
start_ind, end_ind = [fn.nearest_index(r, pt) for pt in (r_start, r_end)]
r = r[start_ind:end_ind]

nl_all = np.loadtxt('nl.dat')
omega_list = np.loadtxt('muhz.dat') * 1e-6 / OM #normlaised frequency list

#the frequency window around each mode
f_window = 50.0 #in muHz

#tolerance of offset of QDPT wet DPT beyond which cross-coupling is important
tol_percent = 5.0   #in percent. tol = 5.0 means 5% tolerance or QPDT rms has to be within 95% of DPT rms.


# l_max = int(np.amax(l_n0) - np.amax(s))
# l_max = int(np.amax(l_n0))
l_max = 30
nmax = 20

#max s values for DR and magnetic field to speed up supermatrix computation
s_max_DR = 5    

#if we want to use smoothened kernels
smoothen = True

#4 components of munu as we ar clubbing together (-- with ++) and (0- with 0+)
qdpt_contrib_rel = np.zeros((30,300))
# qdpt_contrib = np.zeros((4,300))

for n0 in range(0,nmax+1):
    nl_list_n0 = nl_all[nl_all[:,0]==n0]
    l_n0 = nl_list_n0[:,1]

    # l_min = int(np.amin(l_n0) + np.amax(s))
    l_min = int(np.amin(l_n0))
    #running over all the central modes whose frequency shifts we want to find
    for l0 in range(l_min,l_max+1):
        # nl_list = np.zeros((2*np.amax(s)+1,2))  #modes to be considered for a certain central mode
        # nl_list[:,0] = n0   #branch to which it should belong to
        # nl_list[:,1] = np.arange(l0-s_max_H,l0+s_max_H+1) #the +/- s_H_max around l0
        # nl_list = nl_list.astype('int64')   #making the modes integers for future convenience

        omega_nl0 = omega_list[fn.find_nl(n0, l0)]
        # omega_ref0 = np.mean(omega_nl)  #finding the central frequency for doing QDPT
        omega_ref0 = omega_nl0  #setting the delta omegas of the central mode to be around zero

        #central mode is at index zero in nl_list
        nl_list = fn.nearest_freq_modes(l0,s_max_DR,omega_nl0,f_window)
        nl_list = nl_list.astype('int64')   #making the modes integers for future convenience
        
        omega_nl = np.array([omega_list[fn.find_nl(mode[0], mode[1])] for mode in nl_list]) #important to have nl_list as integer type
        nearest_omega_jump = np.sort(np.abs(omega_nl-omega_nl0))[1]

        total_m = len(nl_list) + 2*np.sum(nl_list, axis = 0)[1] #total length of supermatrix

        #MAGNETIC PERTUBATION
        Z_large = np.zeros((total_m, total_m),dtype='complex128')
        Z_diag = np.identity(total_m,dtype='complex128')    #to store the omega_nl**2 - omega_ref0**2
        Z_dpt = np.zeros((2*l0+1, 2*l0+1),dtype='complex128')  #for self-coupling approximation
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
                if(np.abs(l-l_)>s_max_DR): 
                    mj_beg += 2*l+1
                    print('Skipping')
                    continue
                
                #Checking if mode pair coupling has been already computed
                if(os.path.exists('DR_Simulation_submatrices/%i_%i_%i_%i.npy'%(n_,l_,n,l))):
                    print('Computed mode exists')
                    Z_large[mi_beg:mi_end,mj_beg:mj_end] = np.load('DR_Simulation_submatrices/%i_%i_%i_%i.npy'%(n_,l_,n,l))

                #Checking if the flipped modes have been calculated.
                elif(os.path.exists('DR_Simulation_submatrices/%i_%i_%i_%i.npy'%(n,l,n_,l_))):
                    print('Computed flipped mode exists')
                    Z_CT = np.load('DR_Simulation_submatrices/%i_%i_%i_%i.npy'%(n,l,n_,l_))
                    #Performing a hermitian tranpose
                    Z_large[mi_beg:mi_end,mj_beg:mj_end] = np.conjugate(np.transpose(Z_CT,(1,0)))

                #Only if none of the above exists, then we compute them
                else:
                    temp_matrix = submatrix.diffrot(n_,n,l_,l,r,omega_ref0)
                    #print(temp_matrix)
                    temp_matrix = np.diag(temp_matrix)
            #        temp_matrix = np.diag(submatrix.submatrix_diagonal_diffrot(n_,l_,r,omega_nl[i]))
                    del_i, del_j = l_-np.amin([l_,l]), l-np.amin([l_,l])
                    temp_matrix = np.pad(temp_matrix,((del_i,del_i),(del_j,del_j)),'constant',\
                      constant_values = 0)                
                    Z_large[mi_beg:mi_end,mj_beg:mj_end] = temp_matrix  
                    np.save('DR_Simulation_submatrices/%i_%i_%i_%i.npy'%(n_,l_,n,l),Z_large[mi_beg:mi_end,mj_beg:mj_end])                       
            
                mj_beg += 2*l+1

                
            Z_diag[mi_beg:mi_end,mi_beg:mi_end] *= (omega_nl[i]**2 - omega_ref0**2)

            mi_beg += 2*l_+1
    

        Herm_res = Z_large - np.conj(np.transpose(Z_large,(1,0)))
        if(np.amax(np.abs(Herm_res.real)) > 1e-12 or np.amax(np.abs(Herm_res.imag)) > 1e-12): print('Supermatrix is not Hermitian')



        #Analyzing the supermatrix to find the relative contribution of QDPT over DPT

        # Z_dpt = np.load('DR_Simulation_submatrices/%i_%i_%i_%i.npy'%(n0,l0,n0,l0)) #File should exist at this point 
        Z_dpt = np.diag(Z_large)
        #finding the relative cross-coupling contributions for a particular munu 

        title ='Clean'  #Initializing the case as clean. Assuming that the modes are well spaced after splitting
        Z = Z_large[:,:]   

        # eig_vals_dpt,eig_vts_dpt = np.linalg.eigh(Z_dpt)     #Finding the frequency shifts only due to an isolated multiplet
        eig_vals_dpt = np.diag(Z_large)

        #inserting the diagonal component of the supermatrix
        Z += Z_diag   #Adding the diagonal for QDPT
        eig_vals_qdpt,eig_vts_qdpt = np.linalg.eigh(Z)

        m_ind_arr = np.zeros(np.shape(eig_vals_qdpt),dtype=int)

        for i in range(len(eig_vals_qdpt)):
            m_ind_arr[i] = np.argmax(np.abs(eig_vts_qdpt[i]))

        eig_vals_qdpt_arranged = eig_vals_qdpt[m_ind_arr]

        #Initializing variables for analyzing mode spacing after splitting.
        evals_qdpt_diff_abssorted = np.diff(np.sort(np.abs(eig_vals_qdpt)))
        cent_mode_SD = np.sqrt(np.var(evals_qdpt_diff_abssorted[:2*l0]/(2*omega_ref0)))
        freq_jump = evals_qdpt_diff_abssorted[2*l0]/(2*omega_ref0)


        #######################
        #Extracting frequencies

        l_local = 0
        omega_nl_arr = np.zeros(total_m)
        for i in range(len(nl_list)):
            omega_nl_arr[l_local:l_local + 2*nl_list[i,1]+1] = \
                        np.ones(2*nl_list[i,1]+1)*omega_nl[i]
            l_local += 2*nl_list[i,1]+1

        f_dpt = (omega_nl0 + eig_vals_dpt/(2*omega_nl0)) * OM *1e6
        f_qdpt = np.sqrt(omega_ref0**2 + eig_vals_qdpt) * OM *1e6
        f_qdpt_arranged = np.sqrt(omega_ref0**2 + eig_vals_qdpt_arranged) * OM *1e6




        # Checking if we should already discard some modes which come too close or overlap.
        # if(freq_jump <= 0.1*nearest_omega_jump or cent_mode_SD/freq_jump > 0.3):
        #     qdpt_contrib_rel[n0,l0] = 100.0    #set a high value
        #     title ='Unclean'    #Marking them unclean to label the plots

        #     plt.figure()
        #     # plt.plot(np.sort(np.abs(eig_vals_qdpt/(2*omega_ref0))),'.')
        #     plt.plot(f_qdpt_arranged,'.')
        #     plt.plot(f_dpt,'--')
        #     # plt.plot(np.sort(np.abs(omega_nl_arr-omega_ref0))) 
        #     plt.plot(omega_nl_arr*OM*1e6) 
        #     plt.savefig('./DR_coupled_modes_realfrequencies/%i_%i.png'%(n0,l0))
        #     plt.close()

        #     continue    #abandon further analysis
            

        #######################
        #Comparing QDPT and DPT
        #the central mode always comes first because of the algorithm in fn.nearest_freq_modes

        domega_QDPT = np.linalg.norm(f_qdpt_arranged[:2*l0+1] - omega_nl0*OM*1e6)
        domega_DPT = np.linalg.norm(f_dpt[:2*l0+1] - omega_nl0*OM*1e6)

        rel_offset_percent = np.abs((domega_QDPT-domega_DPT)/domega_DPT) * 100.0

        # if(rel_offset_percent <= tol_percent): 
        qdpt_contrib_rel[n0,l0] = rel_offset_percent  

                
inf_ind = np.isinf(qdpt_contrib_rel)
nan_ind = np.isnan(qdpt_contrib_rel)

qdpt_contrib_rel[inf_ind] = 0.0
qdpt_contrib_rel[nan_ind] = 0.0
qdpt_contrib_rel = np.ma.masked_greater(qdpt_contrib_rel,90) 
# qdpt_contrib_rel = np.ma.masked_invalid(qdpt_contrib_rel) 

colors = qdpt_contrib_rel[:nmax+1,:l_max+1]
size = qdpt_contrib_rel[:nmax+1,:l_max+1]/0.01 * 500

size[size<10.0] = 10.0    #Setting the minimum limit to 1.0 => Correspond to 5% or less change

vmin = 0  #To make the smallest dots look gray
vmax = np.amax(colors)

plt.figure(figsize=(12, 10), dpi=200, facecolor='w', edgecolor='k')
for n0 in range(nmax+1):
    nl_list = nl_all[nl_all[:,0]==n0]
    nl_list = nl_list.astype('int64')

    nl_list = nl_list[nl_list[:,1]<l_max+1] #choosing to plot till l_max computed
    l = nl_list[:,1]    #isolating the \ell's

    omega_nl = np.array([omega_list[fn.find_nl(mode[0], mode[1])] for mode in nl_list]) #important to have nl_list as integer type

    plt.scatter(l,omega_nl*OM*1e6/1e3,s=size[n0,l[0]:l[-1]+1], \
                        c=colors[n0,l[0]:l[-1]+1], linewidth=0.5, edgecolor='k',cmap='binary',vmin=vmin,vmax=vmax,alpha = 1.0)

plt.colorbar()

plt.xlim([-1,l_max+1])
plt.ylim([0.490,4.400])

plt.title('$\\frac{L_2^{QDPT}-L_2^{DPT}}{L_2^{DPT}} \\times 100$%',\
                fontsize = 20, pad = 14)
plt.text(0,3.9,'$\Omega(r,\\theta)$',fontsize=18)

plt.xlabel('$\ell$',fontsize=18)
plt.ylabel('Unperturbed frequency $\omega_0$ in mHz',fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('Cross_coupling_goodness/DR_all.pdf')

