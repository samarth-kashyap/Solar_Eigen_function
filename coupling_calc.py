import numpy as np
import matplotlib.pyplot as plt
plt.ion()

nl_list = [(1, 5), (1, 7), (1, 9)]
nl_list = np.array(nl_list)

fname = ''
for i in range(len(nl_list)):
    fname = fname + str(nl_list[i,0]) + '_' + str(nl_list[i,1])
    if(i!=len(nl_list)-1): fname = fname + '-'
         

Z = np.loadtxt('Z_%s.dat'%fname)
Z_diag = np.loadtxt('Z_diag_%s.dat'%fname)
OM = np.loadtxt('../OM.dat')

Z = np.real(Z)
Z_diag = np.real(Z_diag)

n_modes = len(nl_list)

# for i in range(n_modes):
#     diag_no = 2*np.sum(nl_list[0,1]) + 1 + np.abs(nl_list[i][1]-nl_list[0][1]) 
#     inst_diag = np.diag(Z,diag_no)
#     plt.plot(inst_diag)
#     print(diag_no,inst_diag)

for i in range(n_modes):
    for j in range(i+1,n_modes):
        cumul_l = np.sum(nl_list[i:j,1])
        m_skip = np.abs(nl_list[j,1]-nl_list[j-1,1])
        diag_no = 2*cumul_l + j-i + m_skip  
        cross_coupling = np.trim_zeros(np.diag(Z,diag_no))

        if(len(cross_coupling) == 0): continue

        coupling_strength = np.zeros(len(cross_coupling))

        i_inds, j_inds = np.where(np.abs(Z - cross_coupling[0])<1e-6)
        local_i = i_inds[0]
        local_j = j_inds[0]

        for k in range(len(cross_coupling)):
            coupling_strength[k] = cross_coupling[k] \
                                /np.abs(Z[local_i,local_i] - Z[local_j,local_j])
            local_i += 1
            local_j += 1

        plt.plot(coupling_strength * OM *1e6,'.')

        #for the negative diagonal
        diag_no *= -1
        cross_coupling = np.trim_zeros(np.diag(Z,diag_no))

        if(len(cross_coupling) == 0): continue

        coupling_strength = np.zeros(len(cross_coupling))

        i_inds, j_inds = np.where(np.abs(Z - cross_coupling[0])<1e-6)
        local_i = i_inds[0]
        local_j = j_inds[0]

        for k in range(len(cross_coupling)):
            coupling_strength[k] = cross_coupling[k] \
                                /np.abs(Z[local_i,local_i] - Z[local_j,local_j])
            local_i += 1
            local_j += 1

        plt.plot(coupling_strength * OM *1e6,'--')