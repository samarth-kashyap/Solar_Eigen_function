import timing
import numpy as np
import matplotlib.pyplot as plt
import functions as fn
from os import getcwd
import get_kernels as gkerns

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

omega_list = np.loadtxt('muhz.dat')
omega_nl1 = omega_list[fn.find_nl(n1, l1)]
omega_nl2 = omega_list[fn.find_nl(n2, l2)]
omega_nl3 = omega_list[fn.find_nl(n3, l3)]

m = np.array([0])
m_ = np.array([0])
s = np.array([2])

#condition about whether or not to scale by rho
multiplyrho = True

# plot_fac = OM**2 * 1e12 * (4.*np.pi/3) * 1e-10 #unit = muHz G^(-2) V_sol^(-1)
plot_fac = OM**2 * 1e12 * 1e-10 #unit = muHz G^(-2)

#extracting rho in an unclean fashion
rho,__,__,__,__,__,__ = np.array(gkerns.Hkernels(n1,l1,m,n1,l1,m,s,r)\
                        .ret_kerns_axis_symm(a_coeffkerns = True))

#Kernels for a-coefficients for Lorentz stress

# kern = gkerns.Hkernels(n1,l1,m,n1,l1,m,s,r)
__, Bmm1, B0m1,B001, Bpm1,_,_ = np.array(gkerns.Hkernels(n1,l1,m,n1,l1,m,s,r)\
                        .ret_kerns_axis_symm(a_coeffkerns = True))*plot_fac/(-2*omega_nl1)
__, Bmm2, B0m2,B002, Bpm2,_,_ = np.array(gkerns.Hkernels(n2,l2,m,n2,l2,m,s,r).\
                        ret_kerns_axis_symm(a_coeffkerns = True))*plot_fac/(-2*omega_nl2)
__, Bmm3, B0m3,B003, Bpm3,_,_ = np.array(gkerns.Hkernels(n3,l3,m,n3,l3,m,s,r).\
                        ret_kerns_axis_symm(a_coeffkerns = True))*plot_fac/(-2*omega_nl3)


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
plt.figure(num=None, figsize=(11, 8), dpi=dpi, facecolor='w', edgecolor='k')

plt.subplot(221)
plt.plot(r,Bmm1[0,0], label = '('+str(n1)+','+str(l1)+')')
plt.plot(r,Bmm2[0,0], label = '('+str(n2)+','+str(l2)+')')
plt.plot(r,Bmm3[0,0], label = '('+str(n3)+','+str(l3)+')')
plt.xlabel('$r/R_{\odot}$',fontsize=14)
plt.xlim(r_start,r_end)
plt.legend()
plt.grid(True)
# plt.ylabel('$\mathcal{B}^{--}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')
plt.ylabel('$\\rho\mathcal{B}^{--}_{20}$ in $(\mu Hz G^{-2}M_{\odot}V_{\odot}^{-1})$',\
            fontsize = 14)

plt.subplot(222)
plt.plot(r,B0m1[0,0], label = '('+str(n1)+','+str(l1)+')')
plt.plot(r,B0m2[0,0], label = '('+str(n2)+','+str(l2)+')')
plt.plot(r,B0m3[0,0], label = '('+str(n3)+','+str(l3)+')')
plt.xlabel('$r/R_{\odot}$',fontsize=14)
plt.xlim(r_start,r_end)
plt.legend()
plt.grid(True)
# plt.ylabel('$\mathcal{B}^{0-}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')
plt.ylabel('$\\rho\mathcal{B}^{0-}_{20}$ in $(\mu Hz G^{-2}M_{\odot}V_{\odot}^{-1})$',\
            fontsize = 14)

plt.subplot(223)
plt.plot(r,B001[0,0], label = '('+str(n1)+','+str(l1)+')')
plt.plot(r,B002[0,0], label = '('+str(n2)+','+str(l2)+')')
plt.plot(r,B003[0,0], label = '('+str(n3)+','+str(l3)+')')
plt.xlabel('$r/R_{\odot}$',fontsize=14)
plt.xlim(r_start,r_end)
plt.legend()
plt.grid(True)
# plt.ylabel('$\mathcal{B}^{00}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')
plt.ylabel('$\\rho\mathcal{B}^{00}_{20}$ in $(\mu Hz G^{-2}M_{\odot}V_{\odot}^{-1})$',\
            fontsize = 14)

plt.subplot(224)
plt.plot(r,Bpm1[0,0], label = '('+str(n1)+','+str(l1)+')')
plt.plot(r,Bpm2[0,0], label = '('+str(n2)+','+str(l2)+')')
plt.plot(r,Bpm3[0,0], label = '('+str(n3)+','+str(l3)+')')
plt.xlabel('$r/R_{\odot}$',fontsize=14)
plt.xlim(r_start,r_end)
plt.legend()
plt.grid(True)
# plt.ylabel('$\mathcal{B}^{+-}_{20}$ in $(\mu Hz^2 G^{-2}V_{\odot}^{-1})$')
plt.ylabel('$\\rho\mathcal{B}^{+-}_{20}$ in $(\mu Hz G^{-2}M_{\odot}V_{\odot}^{-1})$',\
            fontsize = 14)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.savefig('./kern2.pdf')

plt.show()
