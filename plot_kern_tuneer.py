import timing
import numpy as np
import matplotlib.pyplot as plt
import functions as fn
from os import getcwd
import get_kernels_herm as gkerns
import sys

plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

kernclock = timing.stopclock()
tstamp = kernclock.lap

r = np.loadtxt('r.dat')
r_start = 0.68
r_start_clip = 0.9
r_end_clip = 0.9
r_end = 1.0
start_ind = fn.nearest_index(r,r_start)
end_ind = fn.nearest_index(r,r_end)
r = r[start_ind:end_ind+1]
OM = np.loadtxt('OM.dat')

# n1,l1 = 4,3
# n2,l2 = 1,10
# n3,l3 = 0,60

n1,l1 = 5,110
n2,l2 = 4,60
n3,l3 = 2,10

omega_list = np.loadtxt('muhz.dat')
omega_nl1 = omega_list[fn.find_nl(n1, l1)]
omega_nl2 = omega_list[fn.find_nl(n2, l2)]
omega_nl3 = omega_list[fn.find_nl(n3, l3)]

m = np.array([0])
m_ = np.array([0])
s = np.array([2])

#condition about whether or not to scale by rho
multiplyrho = True
smoothen = True

# plot_fac = OM**2 * 1e12 * (4.*np.pi/3) * 1e-10 #unit = muHz G^(-2) V_sol^(-1)
plot_fac = OM**2 * 1e12 * 1e-10 #unit = muHz G^(-2)

#extracting rho in an unclean fashion
rho,__,__,__,__,__,__ = np.array(gkerns.Hkernels(n1,l1,m,n1,l1,m,s,r)\
                        .ret_kerns_axis_symm(smoothen,a_coeffkerns = True))

#Kernels for a-coefficients for Lorentz stress

# kern = gkerns.Hkernels(n1,l1,m,n1,l1,m,s,r)
__, Bmm1, B0m1,B001, Bpm1,_,_ = np.array(gkerns.Hkernels(n1,l1,m,n1,l1,m,s,r)\
                        .ret_kerns_axis_symm(smoothen,a_coeffkerns = True))*plot_fac/(-2*omega_nl1)
__, Bmm2, B0m2,B002, Bpm2,_,_ = np.array(gkerns.Hkernels(n2,l2,m,n2,l2,m,s,r).\
                        ret_kerns_axis_symm(smoothen,a_coeffkerns = True))*plot_fac/(-2*omega_nl2)
__, Bmm3, B0m3,B003, Bpm3,_,_ = np.array(gkerns.Hkernels(n3,l3,m,n3,l3,m,s,r).\
                        ret_kerns_axis_symm(smoothen,a_coeffkerns = True))*plot_fac/(-2*omega_nl3)


#############################################################################

#Purely lorentz stress kernels

# Bmm1, B0m1,B001, Bpm1,_,_ = np.array(gkerns.Hkernels(n1,l1,m,n1,l1,m,s,r)\
#                         .ret_kerns_axis_symm())*plot_fac
# Bmm2, B0m2,B002, Bpm2,_,_ = np.array(gkerns.Hkernels(n2,l2,m,n2,l2,m,s,r).\
#                         ret_kerns_axis_symm())*plot_fac
# Bmm3, B0m3,B003, Bpm3,_,_ = np.array(gkerns.Hkernels(n3,l3,m,n3,l3,m,s,r).\
#                         ret_kerns_axis_symm())*plot_fac

tstamp('kernel calculation time')

npts = 300   #check the npts in get_kernels
r_new = np.linspace(np.amin(r),np.amax(r),npts)
r = r_new

# rho_i = np.loadtxt('rho.dat')
# rho_i = rho_i[start_ind:end_ind+1]
# rho = np.linspace(rho_i[0],rho_i[-1],npts)

if(multiplyrho==True):
    Bmm1, B0m1,B001, Bpm1 = rho*Bmm1, rho*B0m1,rho*B001, rho*Bpm1
    Bmm2, B0m2,B002, Bpm2 = rho*Bmm2, rho*B0m2,rho*B002, rho*Bpm2
    Bmm3, B0m3,B003, Bpm3 = rho*Bmm3, rho*B0m3,rho*B003, rho*Bpm3

    #plot_fac should have units of  muHz G^(-2) M_sol V_sol^(-1)
    #M_sol/(4pi/3 * R_sol = 1 g/cc)

dpi = 80

# #Making background dark
# fig.patch.set_facecolor('black')
# plt.style.use('dark_background')
# Color keywords were: 'gold', 'red', '--w'

if(len(sys.argv) == 1): #if it does not contain a commandline argument. Default kernels shall be plotted.

    fig = plt.figure(num=None, figsize=(11, 8), dpi=dpi, facecolor='w', edgecolor='k')

    plt.subplot(221)
    plt.plot(Bmm1[0,0],r, label = '('+str(n1)+','+str(l1)+')')
    plt.plot(Bmm2[0,0],r, label = '('+str(n2)+','+str(l2)+')')
    plt.plot(Bmm3[0,0],r, label = '('+str(n3)+','+str(l3)+')')
    plt.ylabel('$r/R_{\odot}$',fontsize=14)
    plt.ylim(r_start,r_end)
    plt.legend()
    plt.grid(True,alpha = 0.2)
    # plt.ylabel('$\mathcal{B}^{--}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')
    plt.xlabel('$\\rho\mathcal{A}^{--}_{20}$',\
                fontsize = 14)

    plt.subplot(222)
    plt.plot(B0m1[0,0],r, label = '('+str(n1)+','+str(l1)+')')
    plt.plot(B0m2[0,0],r, label = '('+str(n2)+','+str(l2)+')')
    plt.plot(B0m3[0,0],r, label = '('+str(n3)+','+str(l3)+')')
    plt.ylabel('$r/R_{\odot}$',fontsize=14)
    plt.ylim(r_start,r_end)
    plt.legend()
    plt.grid(True,alpha=0.2)
    # plt.ylabel('$\mathcal{B}^{0-}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')
    plt.xlabel('$\\rho\mathcal{A}^{0-}_{20}$',\
                fontsize = 14)

    plt.subplot(223)
    plt.plot(B001[0,0],r, label = '('+str(n1)+','+str(l1)+')')
    plt.plot(B002[0,0],r, label = '('+str(n2)+','+str(l2)+')')
    plt.plot(B003[0,0],r, label = '('+str(n3)+','+str(l3)+')')
    plt.ylabel('$r/R_{\odot}$',fontsize=14)
    plt.ylim(r_start,r_end)
    plt.legend()
    plt.grid(True,alpha=0.2)
    # plt.ylabel('$\mathcal{B}^{00}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')
    plt.xlabel('$\\rho\mathcal{A}^{00}_{20}$',\
                fontsize = 14)

    plt.subplot(224)
    plt.plot(Bpm1[0,0],r, label = '('+str(n1)+','+str(l1)+')')
    plt.plot(Bpm2[0,0],r, label = '('+str(n2)+','+str(l2)+')')
    plt.plot(Bpm3[0,0],r, label = '('+str(n3)+','+str(l3)+')')
    plt.ylabel('$r/R_{\odot}$',fontsize=14)
    plt.ylim(r_start,r_end)
    plt.legend()
    plt.grid(True,alpha=0.2)
    # plt.ylabel('$\mathcal{B}^{+-}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')
    plt.xlabel('$\\rho\mathcal{A}^{+-}_{20}$',\
                fontsize = 14)


    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)

    plt.title('$\\rho\mathcal{A}^{\mu\\nu}_{20}$ in $(\mu Hz G^{-2}M_{\odot}V_{\odot}^{-1})$',\
                fontsize = 18, pad = 20)

    plt.savefig('./kern2_new2.pdf',dpi=200)

    plt.show()

else:   #if it contains an argument, kernels shall be zoomed in to show depth sensitivity

    r_end_ind = np.argmin(np.abs(r-r_end_clip))

    fig,ax = plt.subplots(2,4, figsize=(14, 12), dpi=dpi, facecolor='w', edgecolor='k')


    ax[0,0].plot(Bmm1[0,0],r, label = '('+str(n1)+','+str(l1)+')')
    ax[0,0].plot(Bmm2[0,0],r, label = '('+str(n2)+','+str(l2)+')')
    ax[0,0].plot(Bmm3[0,0],r, label = '('+str(n3)+','+str(l3)+')')
    ax[0,0].set_ylabel('$r/R_{\odot}$',fontsize=14)
    ax[0,0].set_ylim(r_start_clip,r_end)
    ax[0,0].legend()
    ax[0,0].grid(True,alpha = 0.2)
    # plt.ylabel('$\mathcal{B}^{--}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')
    ax[0,0].set_xlabel('$\\rho\mathcal{A}^{--}_{20}$',\
                fontsize = 14)


    x_lim_max = np.amax(np.array([np.amax(np.abs(Bmm1[:,:,:r_end_ind])),np.amax(np.abs(Bmm2[:,:,:r_end_ind])),np.amax(np.abs(Bmm3[:,:,:r_end_ind]))]))
    x_lim_max *= 1.1

    ax[1,0].plot(Bmm1[0,0],r, label = '('+str(n1)+','+str(l1)+')')
    ax[1,0].plot(Bmm2[0,0],r, label = '('+str(n2)+','+str(l2)+')')
    ax[1,0].plot(Bmm3[0,0],r, label = '('+str(n3)+','+str(l3)+')')
    ax[1,0].set_ylabel('$r/R_{\odot}$',fontsize=14)
    ax[1,0].set_ylim(r_start,r_end_clip)
    ax[1,0].set_xlim(-x_lim_max,x_lim_max)
    ax[1,0].legend()
    ax[1,0].grid(True,alpha = 0.2)
    # ax[1,0].ylabel('$\mathcal{B}^{--}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')
    ax[1,0].set_xlabel('$\\rho\mathcal{A}^{--}_{20}$',\
                fontsize = 14)

    ax[0,1].plot(B0m1[0,0],r, label = '('+str(n1)+','+str(l1)+')')
    ax[0,1].plot(B0m2[0,0],r, label = '('+str(n2)+','+str(l2)+')')
    ax[0,1].plot(B0m3[0,0],r, label = '('+str(n3)+','+str(l3)+')')
    ax[0,1].set_ylabel('$r/R_{\odot}$',fontsize=14)
    ax[0,1].set_ylim(r_start_clip,r_end)
    ax[0,1].legend()
    ax[0,1].grid(True,alpha=0.2)
    # ax[0,1].ylabel('$\mathcal{B}^{0-}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')
    ax[0,1].set_xlabel('$\\rho\mathcal{A}^{0-}_{20}$',\
                fontsize = 14)

    x_lim_max = np.amax(np.array([np.amax(np.abs(B0m1[:,:,:r_end_ind])),np.amax(np.abs(B0m2[:,:,:r_end_ind])),np.amax(np.abs(B0m3[:,:,:r_end_ind]))]))
    x_lim_max *= 1.1

    ax[1,1].plot(B0m1[0,0],r, label = '('+str(n1)+','+str(l1)+')')
    ax[1,1].plot(B0m2[0,0],r, label = '('+str(n2)+','+str(l2)+')')
    ax[1,1].plot(B0m3[0,0],r, label = '('+str(n3)+','+str(l3)+')')
    ax[1,1].set_ylabel('$r/R_{\odot}$',fontsize=14)
    ax[1,1].set_ylim(r_start,r_end_clip)
    ax[1,1].set_xlim(-x_lim_max,x_lim_max)
    ax[1,1].legend()
    ax[1,1].grid(True,alpha=0.2)
    # ax[1,1].ylabel('$\mathcal{B}^{0-}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')
    ax[1,1].set_xlabel('$\\rho\mathcal{A}^{0-}_{20}$',\
                fontsize = 14)

    ax[0,2].plot(B001[0,0],r, label = '('+str(n1)+','+str(l1)+')')
    ax[0,2].plot(B002[0,0],r, label = '('+str(n2)+','+str(l2)+')')
    ax[0,2].plot(B003[0,0],r, label = '('+str(n3)+','+str(l3)+')')
    ax[0,2].set_ylabel('$r/R_{\odot}$',fontsize=14)
    ax[0,2].set_ylim(r_start_clip,r_end)
    ax[0,2].legend()
    ax[0,2].grid(True,alpha=0.2)
    # ax[0,2].ylabel('$\mathcal{B}^{00}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')
    ax[0,2].set_xlabel('$\\rho\mathcal{A}^{00}_{20}$',\
                fontsize = 14)

    x_lim_max = np.amax(np.array([np.amax(np.abs(B001[:,:,:r_end_ind])),np.amax(np.abs(B002[:,:,:r_end_ind])),np.amax(np.abs(B003[:,:,:r_end_ind]))]))
    x_lim_max *= 1.1

    ax[1,2].plot(B001[0,0],r, label = '('+str(n1)+','+str(l1)+')')
    ax[1,2].plot(B002[0,0],r, label = '('+str(n2)+','+str(l2)+')')
    ax[1,2].plot(B003[0,0],r, label = '('+str(n3)+','+str(l3)+')')
    ax[1,2].set_ylabel('$r/R_{\odot}$',fontsize=14)
    ax[1,2].set_ylim(r_start,r_end_clip)
    ax[1,2].set_xlim(-x_lim_max,x_lim_max)
    ax[1,2].legend()
    ax[1,2].grid(True,alpha=0.2)
    # ax[1,2].ylabel('$\mathcal{B}^{00}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')
    ax[1,2].set_xlabel('$\\rho\mathcal{A}^{00}_{20}$',\
                fontsize = 14)

    ax[0,3].plot(Bpm1[0,0],r, label = '('+str(n1)+','+str(l1)+')')
    ax[0,3].plot(Bpm2[0,0],r, label = '('+str(n2)+','+str(l2)+')')
    ax[0,3].plot(Bpm3[0,0],r, label = '('+str(n3)+','+str(l3)+')')
    ax[0,3].set_ylabel('$r/R_{\odot}$',fontsize=14)
    ax[0,3].set_ylim(r_start_clip,r_end)
    ax[0,3].legend()
    ax[0,3].grid(True,alpha=0.2)
    # ax[0,3].ylabel('$\mathcal{B}^{+-}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')
    ax[0,3].set_xlabel('$\\rho\mathcal{A}^{+-}_{20}$',\
                fontsize = 14)


    x_lim_max = np.amax(np.array([np.amax(np.abs(Bpm1[:,:,:r_end_ind])),np.amax(np.abs(Bpm2[:,:,:r_end_ind])),np.amax(np.abs(Bpm3[:,:,:r_end_ind]))]))
    x_lim_max *= 1.1

    ax[1,3].plot(Bpm1[0,0],r, label = '('+str(n1)+','+str(l1)+')')
    ax[1,3].plot(Bpm2[0,0],r, label = '('+str(n2)+','+str(l2)+')')
    ax[1,3].plot(Bpm3[0,0],r, label = '('+str(n3)+','+str(l3)+')')
    ax[1,3].set_ylabel('$r/R_{\odot}$',fontsize=14)
    ax[1,3].set_ylim(r_start,r_end_clip)
    ax[1,3].set_xlim(-x_lim_max,x_lim_max)
    ax[1,3].legend()
    ax[1,3].grid(True,alpha=0.2)
    # ax[1,3].ylabel('$\mathcal{B}^{+-}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')
    ax[1,3].set_xlabel('$\\rho\mathcal{A}^{+-}_{20}$',\
                fontsize = 14)


    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.45)

    plt.title('$\\rho\mathcal{A}^{\mu\\nu}_{20}$ in $(\mu Hz G^{-2}M_{\odot}V_{\odot}^{-1})$',\
                fontsize = 18, pad = 20)

    plt.savefig('./kern_depth_sens.pdf',dpi=200)

    plt.show()


# Unrotated plots
    # plt.subplot(221)
    # plt.plot(Bmm1[0,0],r, label = '('+str(n1)+','+str(l1)+')')
    # plt.plot(Bmm2[0,0],r, label = '('+str(n2)+','+str(l2)+')')
    # plt.plot(Bmm3[0,0],r, label = '('+str(n3)+','+str(l3)+')')
    # plt.xlabel('$r/R_{\odot}$',fontsize=14)
    # #plt.xlim(r_start,r_end)
    # plt.legend()
    # plt.grid(True)
    # # plt.ylabel('$\mathcal{B}^{--}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')
    # plt.ylabel('$\\rho\mathcal{A}^{--}_{20}$ in $(\mu Hz G^{-2}M_{\odot}V_{\odot}^{-1})$',\
    #             fontsize = 14)

    # plt.subplot(222)
    # plt.plot(r,B0m1[0,0], label = '('+str(n1)+','+str(l1)+')')
    # plt.plot(r,B0m2[0,0], label = '('+str(n2)+','+str(l2)+')')
    # plt.plot(r,B0m3[0,0], label = '('+str(n3)+','+str(l3)+')')
    # plt.xlabel('$r/R_{\odot}$',fontsize=14)
    # plt.xlim(r_start,r_end)
    # plt.legend()
    # plt.grid(True)
    # # plt.ylabel('$\mathcal{B}^{0-}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')
    # plt.ylabel('$\\rho\mathcal{A}^{0-}_{20}$ in $(\mu Hz G^{-2}M_{\odot}V_{\odot}^{-1})$',\
    #             fontsize = 14)

    # plt.subplot(223)
    # plt.plot(r,B001[0,0], label = '('+str(n1)+','+str(l1)+')')
    # plt.plot(r,B002[0,0], label = '('+str(n2)+','+str(l2)+')')
    # plt.plot(r,B003[0,0], label = '('+str(n3)+','+str(l3)+')')
    # plt.xlabel('$r/R_{\odot}$',fontsize=14)
    # plt.xlim(r_start,r_end)
    # plt.legend()
    # plt.grid(True)
    # # plt.ylabel('$\mathcal{B}^{00}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')
    # plt.ylabel('$\\rho\mathcal{A}^{00}_{20}$ in $(\mu Hz G^{-2}M_{\odot}V_{\odot}^{-1})$',\
    #             fontsize = 14)

    # plt.subplot(224)
    # plt.plot(r,Bpm1[0,0], label = '('+str(n1)+','+str(l1)+')')
    # plt.plot(r,Bpm2[0,0], label = '('+str(n2)+','+str(l2)+')')
    # plt.plot(r,Bpm3[0,0], label = '('+str(n3)+','+str(l3)+')')
    # plt.xlabel('$r/R_{\odot}$',fontsize=14)
    # plt.xlim(r_start,r_end)
    # plt.legend()
    # plt.grid(True)
    # # plt.ylabel('$\mathcal{B}^{+-}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')
    # plt.ylabel('$\\rho\mathcal{A}^{+-}_{20}$ in $(\mu Hz G^{-2}M_{\odot}V_{\odot}^{-1})$',\
    #             fontsize = 14)