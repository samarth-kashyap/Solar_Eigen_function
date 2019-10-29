import timing
import numpy as np
import matplotlib.pyplot as plt
import functions as fn
from os import getcwd
import get_kernels_herm as gkerns
plt.ion()

plt.rcParams['xtick.labelsize'] = 3
plt.rcParams['ytick.labelsize'] = 3

lw = 0.5

kernclock = timing.stopclock()
tstamp = kernclock.lap

r = np.loadtxt('r.dat')
r_start = 0.68
r_end = 1.0
start_ind = fn.nearest_index(r,r_start)
end_ind = fn.nearest_index(r,r_end)
r = r[start_ind:end_ind+1]
OM = np.loadtxt('OM.dat')

n1,l1 = 4,3
n2,l2 = 1,10
n3,l3 = 0,60

m = np.array([0])
m_ = np.array([0])
s = np.array([2])

omega_list = np.loadtxt('muhz.dat')

#condition about whether or not to scale by rho
multiplyrho = True
smoothen = True

#extracting rho in an unclean fashion using a random mode
rho,__,__,__,__,__,__ = np.array(gkerns.Hkernels(1,10,0,1,10,0,s,r)\
                        .ret_kerns_axis_symm(smoothen,a_coeffkerns = True))

dpi = 80
# fig = plt.figure(num=None, figsize=(11, 8), dpi=dpi, facecolor='w', edgecolor='k')

nstart = 1

ntotal = 5
ltotal = 11
fig, ax = plt.subplots(ntotal,ltotal,figsize=(11,8))
for ncount in range(ntotal):
    for lcount in range(ltotal):
        n = nstart + ncount
        l = 10 + 10*lcount
        omega_nl = omega_list[fn.find_nl(n, l)]

        # plot_fac = OM**2 * 1e12 * (4.*np.pi/3) * 1e-10 #unit = muHz G^(-2) V_sol^(-1)
        plot_fac = OM**2 * 1e12 * 1e-10 #unit = muHz G^(-2)

        #Kernels for a-coefficients for Lorentz stress

        __, Bmm, B0m,B00, Bpm,__,__ = np.array(gkerns.Hkernels(n,l,m,n,l,m,s,r)\
                                .ret_kerns_axis_symm(smoothen,a_coeffkerns = True))*plot_fac/(-2*omega_nl)

        #############################################################################

        #Purely lorentz stress kernels

        # _,Bmm1, B0m1,B001, Bpm1,_,_ = np.array(gkerns.Hkernels(n1,l1,m,n1,l1,m,s,r)\
        #                         .ret_kerns_axis_symm())*plot_fac
        # _,Bmm2, B0m2,B002, Bpm2,_,_ = np.array(gkerns.Hkernels(n2,l2,m,n2,l2,m,s,r).\
        #                         ret_kerns_axis_symm())*plot_fac
        # _,Bmm3, B0m3,B003, Bpm3,_,_ = np.array(gkerns.Hkernels(n3,l3,m,n3,l3,m,s,r).\
        #                         ret_kerns_axis_symm())*plot_fac

        tstamp('kernel calculation time')

        npts = 300   #check the npts in get_kernels
        r_new = np.linspace(np.amin(r),np.amax(r),npts)
        rplot = r_new

        # rho_i = np.loadtxt('rho.dat')
        # rho_i = rho_i[start_ind:end_ind+1]
        # rho = np.linspace(rho_i[0],rho_i[-1],npts)

        if(multiplyrho==True):
            Bmm, B0m,B00, Bpm = rho*Bmm, rho*B0m,rho*B00, rho*Bpm

            #plot_fac should have units of  muHz G^(-2) M_sol V_sol^(-1)
            #M_sol/(4pi/3 * R_sol = 1 g/cc)

        if(ncount == 0 and lcount ==0):
            lmm, = ax[ncount,lcount].plot(Bmm[0,0],rplot,linewidth = lw)
            l0m, = ax[ncount,lcount].plot(B0m[0,0],rplot,linewidth = lw)
            l00, = ax[ncount,lcount].plot(B00[0,0],rplot,linewidth = lw)
            lpm, = ax[ncount,lcount].plot(Bpm[0,0],rplot,linewidth = lw)

        else:
            ax[ncount,lcount].plot(Bmm[0,0],rplot,linewidth = lw)
            ax[ncount,lcount].plot(B0m[0,0],rplot,linewidth = lw)
            ax[ncount,lcount].plot(B00[0,0],rplot,linewidth = lw)
            ax[ncount,lcount].plot(Bpm[0,0],rplot,linewidth = lw)
        ax[ncount,lcount].set_xlim([-2e-7,2e-7])
        ax[ncount,lcount].set_ylim([0.89,1])
        fig.patch.set_visible(False)
        ax[ncount,lcount].axis('off')
        ax[ncount,lcount].title.set_text('${}_{%i}\mathrm{S}_{%i}$'%(n,l))

    
        if(lcount == 0): 
            ax[ncount,lcount].text(-4e-7,0.9,'$0.9R_{\odot} -$',fontsize=8)            
            # ax[ncount,lcount].text(-4e-7,0.7,'$0.7R_{\odot} -$',fontsize=8)
            

        if(lcount == ltotal-1): 
            ax[ncount,lcount].text(2e-7,0.9,'$-   0.9R_{\odot}$',fontsize=8)
            # ax[ncount,lcount].text(2e-7,0.7,'$-   0.7R_{\odot}$',fontsize=8)
        print(ncount,lcount)

fig.legend((lmm,l0m,l00,lpm),('$\\rho\mathcal{A}^{--}_{n\ell}$','$\\rho\mathcal{A}^{0-}_{n\ell}$', \
        '$\\rho\mathcal{A}^{00}_{n\ell}$','$\\rho\mathcal{A}^{+-}_{n\ell}$'),'lower center',ncol=4)


fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95, hspace=0.5,
                    wspace=0.02)

plt.savefig('kerns_allcompare.pdf',dpi = 200)