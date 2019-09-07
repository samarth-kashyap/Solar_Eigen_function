import numpy as np
import functions as fn
import matplotlib.pyplot as plt
import sys
from matplotlib.patches import Rectangle
plt.ion()

omega_list = np.loadtxt('muhz.dat')
nl_list = np.loadtxt('nl.dat')

OM = np.loadtxt('OM.dat')

om_sorted_ind = np.argsort(omega_list)
ind_om = list(zip(om_sorted_ind, np.take(omega_list,om_sorted_ind)))
ind_om = np.array(ind_om)

nl_arranged = np.zeros((len(ind_om),2))

nl_arranged[:,0] = nl_list[ind_om[:,0].astype(int),0]
nl_arranged[:,1] = nl_list[ind_om[:,0].astype(int),1]

l_arranged_by_freq = np.zeros((30,301))
freq_diff = np.zeros((30,300))

for n0 in range(0,30):
    ind_n0 = np.where(nl_arranged[:,0] == n0)
    ind_om_n0 = ind_om[ind_n0]
    # plt.plot(ind_om_n0[:,1])
    #plt.plot(np.diff(ind_om_n0[:,1]),'.')
    l_arranged_by_freq[n0,:len(ind_om_n0)] = nl_arranged[ind_n0,1][0]
    freq_diff[n0,:len(ind_om_n0)-1] = np.diff(ind_om_n0[:,1])

freq_diff = freq_diff

plt.pcolormesh(freq_diff,vmax=20)
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
plt.ylabel('$n$',fontsize=14)
plt.xlabel('$\omega \longrightarrow$',fontsize=14)
plt.title('Colormap of $\omega_{n,k+1} - \omega_{n,k}$; where $\omega_{n,k}$ = sorted $\omega_{n,\ell}$ (fixed $n$)',pad = 20)
plt.colorbar()

plt.savefig('./Coupling_Strength/domega.eps')


# #PLotting the l-variation with increasin frequency
# dpi = 80
# fig = plt.figure(num=None, figsize=(10, 20), dpi=dpi, facecolor='w', edgecolor='k')

# plt.subplot(311)
# plt.pcolormesh(l_arranged_by_freq)
# plt.colorbar(orientation='horizontal',aspect=80)
# plt.ylabel('$n$',fontsize=14)
# plt.xlabel('$\omega \longrightarrow$',fontsize=14)
# plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)

# currentAxis = plt.gca()
# currentAxis.add_patch(Rectangle((1, 0.1), 80, 29.7, edgecolor="red",linewidth=4,fill=False,alpha=0.8))

# plt.title('Colormap showing $\ell$ values with increasing frequency.',fontsize=14,pad = 17)

# plt.subplot(312)
# plt.pcolormesh(l_arranged_by_freq[:,0:80])
# plt.colorbar(orientation='horizontal',aspect=80)
# plt.ylabel('$n$',fontsize=14)
# plt.xlabel('$\omega \longrightarrow$',fontsize=14)
# plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
# plt.title('Zoomed-in colormap')

# plt.plot(np.arange(0,81,1),5.5*np.ones(81),'red',label='$n = 5$')
# plt.plot(np.arange(0,81,1),14.5*np.ones(81),label='$n = 14$')

# plt.legend()

# plt.subplot(313)

# plt.plot(np.arange(0,80,1),l_arranged_by_freq[5,0:80],'ro',label='$n = 5$')
# plt.plot(np.arange(0,80,1),l_arranged_by_freq[14,0:80],'.',label='$n = 14$')
# plt.ylabel('$\ell$',fontsize=14)
# plt.xlabel('$\omega \longrightarrow$',fontsize=14)
# plt.xlim([0,80])
# plt.title('Distribution of $\ell$ for two modes with increasing frequency.', fontsize=14)
# plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)

# plt.legend()

# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.09, right=0.92, hspace=0.05,
#                     wspace=0.35)

# plt.savefig('./Coupling_Strength/l_distribution.eps')
# # plt.show()