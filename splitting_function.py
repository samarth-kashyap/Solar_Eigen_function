import numpy as np
import scipy.special as sp 
import h_components as hcomps
import get_kernels_herm as gkerns
import functions as fn
import scipy.integrate
import matplotlib.pyplot as plt 
import random
plt.ion()

#Modes whose splitting function is to be computed
n,l = 11,8
n_,l_ = 20,10

#adjusting the radius
r = np.loadtxt('r.dat')
r_start, r_end = 0.,1.
start_ind, end_ind = [fn.nearest_index(r, pt) for pt in (r_start, r_end)]
r = r[start_ind:end_ind]

#Lorentz stress components' order
s = np.array([2,4,7,9,11])
smax = np.amax(s)       #value of max s in B
s_H_max = 2*smax         #triangle law 0 <= s_H <= 2s;    s_H_max is max value of s for H 

om = np.vectorize(fn.omega,otypes=[float])
wig_val = np.vectorize(fn.wig,otypes=[float])

#B magnitude profile as a function of radius which is simply squared to get H magnitude for different components
def B0_st_profile(r):
    beta = 1e-4/r**3  #10G on surface
    #1e5 Gauss at tachocline
    beta += np.exp(-0.5*((r-0.7)/0.01)**2)
    #1e7 Gauss at core
    beta += 100*np.exp(-0.5*(r/0.1)**2)
    return beta

#function to return the B-component of all s,t
def ret_Bcomps(s,r):
    B0_st = B0_st_profile(r)
    #Implementing the solenoidal condition
    gradr_B0st = (np.gradient(r**2*B0_st,r)/r)[np.newaxis,:]/om(s,0)[:,np.newaxis]      #shape s x r
    rand_factors = np.zeros((len(s),2*smax+1))
    one_factors = np.zeros((len(s),2*smax+1))
    for s0 in range(len(s)):
        scurrent = s[s0]
        rand_factors[s0,smax-scurrent:smax+scurrent+1] = np.array([random.random() for i in range(2*scurrent+1)]) #shape s x t
        one_factors[s0,smax-scurrent:smax+scurrent+1] = np.array([1.0 for i in range(2*scurrent+1)]) #shape s x t

    B0_st = B0_st[np.newaxis,:]*one_factors[:,:,np.newaxis]
    gradr_B0st = gradr_B0st[:,np.newaxis,:]*one_factors[:,:,np.newaxis]
    Bp_st = rand_factors[:,:,np.newaxis]*gradr_B0st    #shape s x t x r
    Bm_st = gradr_B0st - Bp_st

    B_st_total = np.zeros((3,len(s),2*smax+1,len(r)))
    B_st_total[0] = Bm_st
    B_st_total[1] = B0_st
    B_st_total[2] = Bp_st
    return B_st_total

def ret_Hcomps(mu,nu,s0,t0):
    #computing the brute force way
#     H_st = np.zeros(len(r))
#     for s1 in s:
#         for s2 in s:
#             for t1 in range(-s1,s1+1):
#                 for t2 in range(-s2,s2+1):
#                 #     print(s1,s2,t1)
#                     H_st_temp = B_st[mu,s1-1,t1+smax,:]*B_st[nu,s2-1,t2+smax,:]
#                     H_st_temp = H_st_temp * (-1)**(np.abs(mu+nu+t0)) * np.sqrt((2*s0+1.)*(2*s1+1.)*(2*s2+1.)/(4.* np.pi))
#                     H_st_temp = H_st_temp * wig_val(s0,s1,s2,-t0,t1,t2) * wig_val(s0,s1,s2,-(mu+nu),mu,nu)
#                     H_st += H_st_temp

    #computing using numpy array handling
    H_st_mat = B_st[mu,:,:,np.newaxis,np.newaxis,:]*B_st[nu,np.newaxis,np.newaxis,:,:,:]
    t = np.arange(-smax,smax+1)
    ss1,tt1,ss2,tt2 = np.meshgrid(s,t,s,t,indexing='ij')

    prefactor = (-1)**(np.abs(mu+nu+t0)) * np.sqrt((2*s0+1.)*(2*ss1+1.)*(2*ss2+1.)/(4.* np.pi))
    prefactor *= wig_val(s0,ss1,ss2,-t0,tt1,tt2) * wig_val(s0,ss1,ss2,-(mu+nu),mu,nu)

    H_st_mat = H_st_mat * prefactor[:,:,:,:,np.newaxis]
    H_st_mat = np.sum(H_st_mat,axis=(0,1,2,3))

    return H_st_mat


#getting the kernel components


m = np.arange(-l,l+1,1)    #-l<=m<=l
m_ = np.arange(-l_,l_+1,1)  #-l_<=m<=l_ 

s_H_arr = np.arange(s_H_max+1)[::5]

kern = gkerns.Hkernels(n_,l_,m_,n,l,m,s_H_arr,r,False)
Bmm,B0m,B00,Bpm,Bp0,Bpp = kern.ret_splittingfunction_terms(smoothen=True)

npts = 300   #check the npts in get_kernels
r_new = np.linspace(np.amin(r),np.amax(r),npts)
r = r_new

B_st = ret_Bcomps(s,r)     #computes all components of all st of 

#finding the H-components
Hmm_str = np.zeros((len(s_H_arr),2*s_H_max+1,len(r)))
H0m_str = np.zeros((len(s_H_arr),2*s_H_max+1,len(r)))
H00_str = np.zeros((len(s_H_arr),2*s_H_max+1,len(r)))
Hpm_str = np.zeros((len(s_H_arr),2*s_H_max+1,len(r)))
H0p_str = np.zeros((len(s_H_arr),2*s_H_max+1,len(r)))
Hpp_str = np.zeros((len(s_H_arr),2*s_H_max+1,len(r)))

print('Here')

for s0 in range(len(s_H_arr)):    
#     for t0 in range(-s0,0):
#             Hmm_str[s0,t0+s_H,:]  = ret_Hcomps(-1,-1,s0,t0)
#             H0m_str[s0,t0+s_H,:]  = ret_Hcomps(0,-1,s0,t0)
#             H00_str[s0,t0+s_H,:]  = ret_Hcomps(0,0,s0,t0)
#             Hpm_str[s0,t0+s_H,:]  = ret_Hcomps(1,-1,s0,t0)

#             #Imposing the realness of H along with its symmetry in mu,nu
#             Hpp_str[s0,-t0+s_H,:] = (-1)**(np.abs(t0)) * np.conj(Hmm_str[s0,t0+s_H,:])
#             H0p_str[s0,-t0+s_H,:] = (-1)**(np.abs(t0)) * np.conj(H0m_str[s0,t0+s_H,:])

#             H00_str[s0,-t0+s_H,:] = (-1)**(np.abs(t0)) * np.conj(H00_str[s0,t0+s_H,:])
#             Hpm_str[s0,-t0+s_H,:] = (-1)**(np.abs(t0)) * np.conj(Hpm_str[s0,t0+s_H,:])

#     for t0 in range(0,s0+1):
#             Hmm_str[s0,t0+s_H,:]  = ret_Hcomps(-1,-1,s0,t0)
#             H0m_str[s0,t0+s_H,:]  = ret_Hcomps(0,-1,s0,t0)

#             #Imposing the realness of H along with its symmetry in mu,nu
#             Hpp_str[s0,-t0+s_H,:] = (-1)**(np.abs(t0)) * np.conj(Hmm_str[s0,t0+s_H,:])
#             H0p_str[s0,-t0+s_H,:] = (-1)**(np.abs(t0)) * np.conj(H0m_str[s0,t0+s_H,:])
    
    scurrent = s_H_arr[s0]
    for t0 in range(-scurrent,scurrent+1):
            Hmm_str[s0,t0+s_H_max,:] = ret_Hcomps(-1,-1,scurrent,t0)
            H0m_str[s0,t0+s_H_max,:] = ret_Hcomps(0,-1,scurrent,t0)
            H00_str[s0,t0+s_H_max,:] = ret_Hcomps(0,0,scurrent,t0)
            Hpm_str[s0,t0+s_H_max,:] = ret_Hcomps(1,-1,scurrent,t0)
            Hpp_str[s0,t0+s_H_max,:] = ret_Hcomps(1,1,scurrent,t0)
            H0p_str[s0,t0+s_H_max,:] = ret_Hcomps(0,1,scurrent,t0)

            
    print(scurrent,t0)

#plotting on Mollweide projection
theta = np.linspace(-np.pi/2,np.pi/2,360)
phi = np.linspace(-np.pi,np.pi,720)

thth,phph = np.meshgrid(theta,phi,indexing='ij')

#find integrand by summing all component
Lambda_str = Hpp_str*Bpp[:,np.newaxis,:] + H00_str*B00[:,np.newaxis,:] + Hmm_str*Bmm[:,np.newaxis,:] \
        + 2*Hpm_str*Bpm[:,np.newaxis,:] + 2*H0m_str*B0m[:,np.newaxis,:] + 2*H0p_str*Bp0[:,np.newaxis,:]

#radial integral
Lambda_st = scipy.integrate.trapz(Lambda_str*(r**2),x=r,axis=2)

z = np.zeros(np.shape(thth),dtype='complex128')

fig = plt.figure()
ax = fig.add_subplot(111, projection='aitoff')

for s0 in range(len(s_H_arr)):
    scurrent = s_H_arr[s0]
    for t0 in range(-scurrent,scurrent+1):
        z += Lambda_st[s0,t0+s_H_max]*sp.sph_harm(t0,scurrent,phph+np.pi,thth+np.pi/2.0)
        
im = ax.pcolormesh(phph,thth,np.real(z), cmap=plt.cm.jet)
#  plt.colorbar()  

# z = np.zeros(np.shape(thth),dtype='complex128')
# count = 0
# for s in range(20):
#         for t in range(-s,s+1):
#                 z += random.random()*sp.sph_harm(t,s,phph+np.pi,thth+np.pi/2)
#                 count += 1


