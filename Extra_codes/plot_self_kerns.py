import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.ion()

s = 2

Bmm = np.loadtxt('Bmm_all_s%i.dat'%s)
B0m = np.loadtxt('B0m_all_s%i.dat'%s)
B00 = np.loadtxt('B00_all_s%i.dat'%s)
Bpm = np.loadtxt('Bpm_all_s%i.dat'%s)

ytickinfo = np.loadtxt('ytickinfo.dat')

labels,location = ytickinfo[:,0],ytickinfo[:,1]
labels = labels.astype(int)

Bmm_vmax = np.amax(Bmm)/5
Bmm_vmin = np.amin(Bmm)/5
B0m_vmax = np.amax(B0m)/5
B0m_vmin = np.amin(B0m)/5
B00_vmax = np.amax(B00)/5
B00_vmin = np.amin(B00)/5
Bpm_vmax = np.amax(Bpm)/5
Bpm_vmin = np.amin(Bpm)/5

r_start = 0.68
r_end = 1.0
npts = 300

r = np.linspace(r_start,r_end,npts)

rr,cnl = np.meshgrid(np.linspace(0,len(Bmm),len(Bmm)),r,indexing='ij')

#fig, ax = plt.subplots(2, 2, sharey = True, sharex = True,figsize=(10,8))
fig, ax = plt.subplots(2, 2, sharex = True,figsize=(10,8))

nrg = 100 #no of grid in r from end. Basically array[:,-nrg:]

xlabel, xloc = r[-nrg:], np.linspace(0,nrg-1,nrg)
xlabel = np.round(xlabel,3)

# ax[0,0].pcolormesh(rr[:,-nrg:],cnl[:,-nrg:],Bmm[:,-nrg:],vmin=Bmm_vmin,vmax=Bmm_vmax)
# ax[0,1].pcolormesh(rr[:,-nrg:],cnl[:,-nrg:],B0m[:,-nrg:],vmin=B0m_vmin,vmax=B0m_vmax)
# ax[1,0].pcolormesh(rr[:,-nrg:],cnl[:,-nrg:],B00[:,-nrg:],vmin=B00_vmin,vmax=B00_vmax)
# ax[1,1].pcolormesh(rr[:,-nrg:],cnl[:,-nrg:],Bpm[:,-nrg:],vmin=Bpm_vmin,vmax=Bpm_vmax)

im1 = ax[0,0].pcolormesh(Bmm[:,-nrg:],vmin=Bmm_vmin,vmax=Bmm_vmax)
plt.colorbar(im1,ax=ax[0,0],orientation='vertical',aspect = 40,pad=0.02)
im2 = ax[0,1].pcolormesh(B0m[:,-nrg:],vmin=B0m_vmin,vmax=B0m_vmax)
plt.colorbar(im2,ax=ax[0,1],orientation='vertical',aspect = 40,pad=0.02)
im3 = ax[1,0].pcolormesh(B00[:,-nrg:],vmin=B00_vmin,vmax=B00_vmax)
plt.colorbar(im3,ax=ax[1,0],orientation='vertical',aspect = 40,pad=0.02)
im4 = ax[1,1].pcolormesh(Bpm[:,-nrg:],vmin=Bpm_vmin,vmax=Bpm_vmax)
plt.colorbar(im4,ax=ax[1,1],orientation='vertical',aspect = 40,pad=0.02)
#fig.colorbar(im4,orientation='vertical')

plt.setp(ax, yticks=location[:-1:5],yticklabels=labels[:-1:5])

ax[0,0].text(1.5,4000,"$(a) \\rho \mathcal{A}^{--}_{%i0}$"%s,color='white',fontsize=12)
ax[0,1].text(1.5,4000,"$(b) \\rho \mathcal{A}^{0-}_{%i0}$"%s,color='white',fontsize=12)
ax[1,0].text(1.5,4000,"$(c) \\rho \mathcal{A}^{00}_{%i0}$"%s,color='white',fontsize=12)
ax[1,1].text(1.5,4000,"$(d) \\rho \mathcal{A}^{+-}_{%i0}$"%s,color='black',fontsize=12)

plt.sca(ax[1, 1])
plt.yticks(location[:-1:5], labels[:-1:5])
plt.xticks(xloc[::15],xlabel[::15])

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

plt.ylabel('Cumulative $n_l$',fontsize=14, labelpad = 5.5)
plt.xlabel('$r/R_{\odot}$',fontsize=14)
plt.title('$\\rho \mathcal{A}_{%i0}^{\mu\\nu}$ in $\mu Hz G^{-2}M_{\odot}V_{\odot}^{-1}$'%s,pad=14.5)
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.98, hspace=0.15,
                    wspace=0.02)

plt.savefig('./Bmunu_self_s%i.png'%s,dpi=400)