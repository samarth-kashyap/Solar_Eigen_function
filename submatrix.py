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

def lorentz_diagonal(n_,n,l_,l,r,field_type = 'dipolar',smoothen = False): 
    m = np.arange(-min(l,l_),min(l,l_)+1,1)    #-l<=m<=l
    s = np.array([0,1,2])
    s0 = 1
    r_new = r

    #transition radii for mixed field type
    R1 = 0.75
    R2 = 0.78

    if(smoothen == True):
        npts = 300      #should be less than the len(r) in r.dat
        r_new = np.linspace(np.amin(r),np.amax(r),npts)

    B_mu_r = fn.getB_comps(s0,r_new,R1,R2,field_type)[:,s0,:] #choosing t = 0 comp
    get_h = hcomps.getHcomps(s,m,m,s0,np.array([s0]),r_new,B_mu_r)
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
    Bmm,B0m,B00,Bpm,Bp0,Bpp = kern.ret_kerns_axis_symm(smoothen = smoothen)
    #sys.exit()

    #find integrand by summing all component
    Lambda_sr = hpp[np.newaxis,:,:]*Bpp + h00[np.newaxis,:,:]*B00 + hmm[np.newaxis,:,:]*Bmm \
            + 2*hpm[np.newaxis,:,:]*Bpm + 2*h0m[np.newaxis,:,:]*B0m + 2*hp0[np.newaxis,:,:]*Bp0

    #summing over s before carrying out radial integral
    Lambda_r = np.sum(Lambda_sr,axis=1)

    if(smoothen==True):
        r = r_new

    #radial integral
    Lambda = scipy.integrate.trapz(Lambda_r*(r**2)[np.newaxis,:],x=r,axis=1)
    
    return Lambda

def lorentz_all_st_equalB(n_,n,l_,l,r,s = np.array([0,1,2]),t = np.array([0])):
    m = np.arange(-l,l+1)
    m_ = np.arange(-l_,l_+1)
    mm_,mm = np.meshgrid(m_,m,indexing='ij')

    kern = gkerns.Hkernels(n_,l_,m_,n,l,m,s,r,False)   

    Bmm,B0m,B00,Bpm,Bp0,Bpp = kern.ret_kerns()

    Bmm_t = np.zeros(Bmm.shape)
    B0m_t = np.zeros(B0m.shape)
    B00_t = np.zeros(B00.shape)
    Bpm_t = np.zeros(Bpm.shape)
    Bp0_t = np.zeros(Bp0.shape)
    Bpp_t = np.zeros(Bpp.shape)
    for i in t:
        Bmm_t[mm_-mm==i] += Bmm[mm_-mm==i]
        B0m_t[mm_-mm==i] += B0m[mm_-mm==i]
        B00_t[mm_-mm==i] += B00[mm_-mm==i]
        Bpm_t[mm_-mm==i] += Bpm[mm_-mm==i]
        Bp0_t[mm_-mm==i] += Bp0[mm_-mm==i]
        Bpp_t[mm_-mm==i] += Bpp[mm_-mm==i]


    #Construct h_{st}^{\mu\nu}(r) which is the same for all s,t,\mu,\nu

    b_r = 1e-4/r**3  #10G on surface
    #1e5 Gauss at tachocline
    b_r += np.exp(-0.5*((r-0.7)/0.01)**2)
    #1e7 Gauss at core
    b_r += 100*np.exp(-0.5*(r/0.1)**2)
    h_r = b_r*b_r

    Lambda = np.zeros((6,len(m_),len(m),len(s)))

    #Integrating over r but still retaining dimension of s
    Lambda[0] = scipy.integrate.trapz(Bmm_t*(h_r*(r**2)),x=r,axis=3)
    Lambda[1] = scipy.integrate.trapz(B0m_t*(h_r*(r**2)),x=r,axis=3)
    Lambda[2] = scipy.integrate.trapz(B00_t*(h_r*(r**2)),x=r,axis=3)
    Lambda[3] = scipy.integrate.trapz(Bpm_t*(h_r*(r**2)),x=r,axis=3)
    Lambda[4] = scipy.integrate.trapz(Bp0_t*(h_r*(r**2)),x=r,axis=3)
    Lambda[5] = scipy.integrate.trapz(Bpp_t*(h_r*(r**2)),x=r,axis=3)

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
    
    tstamp()

    C = scipy.integrate.trapz((w*T_kern)*(rho*(r**2))[np.newaxis,:],x=r,axis=1)
    C = C[np.newaxis,:] * (2*((-1)**np.abs(mm))*omega_ref*wig_calc(l_,ss,l,-mm,0,mm))
    C *= np.sqrt((2*l+1) * (2*l_+1) * (2*ss+1)/(4.*np.pi))
    C = np.sum(C, axis = 1)

    return C

