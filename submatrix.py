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

OM = np.loadtxt('OM.dat')
R_sol = 6.956e10 #cm
        
def lorentz(n_,n,l_,l,r,beta =0., field_type = 'dipolar'):   
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


    kern = gkerns.Hkernels(n_,l_,m_,n,l,m,s,r,False)
    Bmm,B0m,B00,Bpm,Bp0,Bpp = kern.ret_kerns()
    #sys.exit()

    #find integrand by summing all component
    Lambda_sr = hpp*Bpp + h00*B00 + hmm*Bmm \
            + 2*hpm*Bpm + 2*h0m*B0m + 2*hp0*Bp0

    #summing over s before carrying out radial integral
    Lambda_r = np.sum(Lambda_sr,axis=2)

    #radial integral
    Lambda = scipy.integrate.trapz(Lambda_r*(r**2)[np.newaxis,:],x=r,axis=2)

    return Lambda

def lorentz_diagonal(n_,n,l_,l,r, field_type = 'dipolar'):   
    m = np.arange(-min(l,l_),min(l,l_)+1,1)    #-l<=m<=l
    s = np.array([0,1,2])
    s0 = 1
    #transition radii for mixed field type
    R1 = r[-15]
    R2 = r[-10]
    B_mu_r = fn.getB_comps(s0,r,R1,R2,field_type)[:,s0,:] #choosing t = 0 comp
    get_h = hcomps.getHcomps(s,m,m,s0,np.array([s0]),r,B_mu_r)
    tstamp()

    H_super = get_h.ret_hcomps_axis_symm()  #- sign due to i in B 

    tstamp('Computed H-components in')

    #distributing the components
    hmm = H_super[0,0]
    h0m = H_super[1,0]
    h00 = H_super[1,1]
    hpm = H_super[2,0]
    hp0 = H_super[2,1]
    hpp = H_super[2,2]

    kern = gkerns.Hkernels(n_,l_,m,n,l,m,s,r,True)
    Bmm,B0m,B00,Bpm,Bp0,Bpp = kern.ret_kerns_axis_symm()
    #sys.exit()

    #find integrand by summing all component
    Lambda_sr = hpp[np.newaxis,:,:]*Bpp + h00[np.newaxis,:,:]*B00 + hmm[np.newaxis,:,:]*Bmm \
            + 2*hpm[np.newaxis,:,:]*Bpm + 2*h0m[np.newaxis,:,:]*B0m + 2*hp0[np.newaxis,:,:]*Bp0

    #summing over s before carrying out radial integral
    Lambda_r = np.sum(Lambda_sr,axis=1)

    #radial integral
    Lambda = scipy.integrate.trapz(Lambda_r*(r**2)[np.newaxis,:],x=r,axis=1)
    
    return Lambda
    
def diffrot(n_,n,l_,l,r,omega_ref,s=np.array([1,3,5])):
    wig_calc = np.vectorize(fn.wig)
    
    r_full = np.loadtxt('r.dat')
    r_start, r_end = np.argmin(np.abs(r_full-r[0])),np.argmin(np.abs(r_full-r[-1]))+1
    rho = np.loadtxt('rho.dat')[r_start:r_end]
    
    m = np.arange(-min(l,l_),min(l,l_)+1,1)    #-l<=m<=l
    m_ = m  #-l_<=m<=l_    
    
    mm,ss = np.meshgrid(m,s,indexing='ij')
    
    kern = gkerns.Hkernels(n_,l_,m,n,l,m,s,r)
    T_kern = kern.Tkern(s)
    
    w = np.loadtxt('w.dat')[:,r_start:r_end]

    C = np.zeros(mm.shape)
    
    C = (2*((-1)**np.abs(mm))*omega_ref*wig_calc(l_,ss,l,-mm,0,mm))[:,:,np.newaxis]\
            *rho[np.newaxis,np.newaxis,:]*(w*T_kern)[np.newaxis,:,:]
    C = np.sqrt((2*l+1) * (2*l_+1) * (2*ss+1)/(4.*np.pi))[:,:,np.newaxis] * C            
    C = np.sum(C, axis = 1)
    C = scipy.integrate.trapz(C*(r**2)[np.newaxis,:],x=r,axis=1)

    return C

