import numpy as np
import functions as fn
import scipy.integrate
import get_kernels as gkerns
import h_components as hcomps
import sys
import math
#code snippet for timing the code
import timing
clock2 = timing.stopclock()
tstamp = clock2.lap

        
def submatrix(n_,n,l_,l,r,beta =0., field_type = 'dipolar'):   
    m = np.arange(-l,l+1,1)    #-l<=m<=l
    m_ = np.arange(-l_,l_+1,1)  #-l_<=m<=l_    
    s = np.array([0,1,2])
    s0 = 1  
    t0 = np.arange(-s0,s0+1)                                       
    #transition radii for mixed field type
    R1 = r[0]
    R2 = r[-1]
    B_mu_t_r = fn.getB_comps(s0,r,R1,R2,field_type)

    get_h = hcomps.getHcomps(s,m_,m,s0,t0,r,B_mu_t_r, beta)

    tstamp()

    H_super = get_h.ret_hcomps()  #- sign due to i in B

    tstamp('Computed H-components in')

    #distributing the components
    hmm = H_super[0,0,:,:,:,:]
    h0m = H_super[1,0,:,:,:,:]
    h00 = H_super[1,1,:,:,:,:]
    hpm = H_super[2,0,:,:,:,:]
    hp0 = H_super[2,1,:,:,:,:]
    hpp = H_super[2,2,:,:,:,:]


    kern = gkerns.Hkernels(n_,l_,m_,n,l,m,s,r)
    Bmm,B0m,B00,Bpm,Bp0,Bpp = kern.ret_kerns()
    #sys.exit()

    #find integrand by summing all component
    Lambda_sr = hpp*Bpp + h00*B00 + hmm*Bmm \
            + 2*hpm*Bpm + 2*h0m*B0m + 2*hp0*Bp0

    #summing over s before carrying out radial integral
    Lambda_r = np.sum(Lambda_sr,axis=2)

    #radial integral
    Lambda = scipy.integrate.trapz(Lambda_r*(r**2)[np.newaxis,:],x=r,axis=2)


    Lambda = np.real(Lambda)
    
    return Lambda
