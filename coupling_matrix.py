import numpy as np
import functions as fn
import matplotlib.pyplot as plt
import scipy.integrate
import get_kernels as gkerns
import h_components as hcomps
plt.ion()
#code snippet for timing the code
import timing
clock2 = timing.stopclock()
tstamp = clock2.lap

class cp_mat_nn_ll_:

    def __init__(self,n,l,n_,l_):
        self.n = n
        self.n_ = n_
        self.l = l
        self.l_ = l_

    def b_radial(self,r,r_cen):
        return np.exp(-0.5*((r-r_cen)/0.0001)**2)

    def ret_coupled_nn_ll_(self):
        #loading and chopping r grid
        r = np.loadtxt('r.dat')
        rpts = 700
        r = r[-rpts:]

        #variable set in stone
        mu = np.array([-1,0,1])
        nu = np.array([-1,0,1])


        #Choosing n,l and n_,l_
        n,l = self.n,self.l
        n_,l_ = self.n_,self.l_

        #mode for toroidal magnetic field
        l_b = 1
        m_b = 0

        r_cen = 0.998
        b_r = b_radial(r,r_cen)

        #Choosing s
        smin = 0
        #smax = 'max_s' for 0 <= s <= 2l+1
        #smax = False for s = s_arr (custom choice)
        #smax = <some integral value> 
        smax = 0
        #custom made s array
        s = np.array([0,1,2,3,4])


        m = np.arange(-l,l+1,1)   # -l<=m<=l
        m_ = np.arange(-l_,l_+1,1) # -l_<=m<=l_


        #For the current case where l=l_,n=n_
        kern_eval = gkerns.Hkernels(l,l_,r,rpts)

        Bmm,B0m,B00,Bpm,Bpp,B0p = kern_eval.isol_multiplet(n,l,s)

        tstamp('kernel evaluated')

        #Fetching the H-components
        get_h = hcomps.getHcomps(s,m,l_b,m_b,r,b_r)

        tstamp()
        H_super = get_h.ret_hcomps()  #this is where its getting computed
        tstamp('Computed H-components in')

        #distributing the components
        hmm = H_super[:,:,0,0,:,:]
        h0m = H_super[:,:,1,0,:,:]
        h00 = H_super[:,:,1,1,:,:]
        hp0 = H_super[:,:,2,1,:,:]
        hpp = H_super[:,:,2,2,:,:]
        hpm = H_super[:,:,2,0,:,:]

        #find integrand by summing all component
        cp_mat_s = hpp*Bpp + h00*B00 + hmm*Bmm \
                + 2*hpm*Bpm + 2*h0m*B0m + 2*hp0*B0p

        #summing over s before carrying out radial integral
        cp_mat_befint = np.sum(cp_mat_s,axis=2)

        #radial integral
        cp_mat = scipy.integrate.trapz(cp_mat_befint,x=r,axis=2)

        plt.pcolormesh(cp_mat)
        plt.colorbar()
        plt.show('Block')

        return cp_mat



