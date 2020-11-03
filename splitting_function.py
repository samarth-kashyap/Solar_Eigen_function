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
n,l = 4,9
# n,l = 4,19
# n,l = 4,7
# n_,l_ = 3,8
n_,l_ = 4,9


#finding the central frequency with which we shall normalize the splitting function
OM = np.loadtxt('OM.dat') #importing normalising frequency value from file (in Hz (cgs))
omega_list = np.loadtxt('muhz.dat') * 1e-6 / OM #normlaised for mode in nl_list]) frequency list
#taking a geometric mean of the frequencies of coupling modes
omega_0 = np.sqrt(omega_list[fn.find_nl(n,l)]*omega_list[fn.find_nl(n_,l_)])

#adjusting the radius
r = np.loadtxt('r.dat')
r_start, r_end = 0.,1.
start_ind, end_ind = [fn.nearest_index(r, pt) for pt in (r_start, r_end)]
r = r[start_ind:end_ind]

#Lorentz stress components' order
s = np.array([1,2,3])
smax = np.amax(s)       #value of max s in B
t_smax = np.arange(-smax,smax+1)
s_H_max = 2*smax         #triangle law 0 <= s_H <= 2s;    s_H_max is max value of s for H 
s_H_arr = np.arange(s_H_max+1)
t_sHmax = np.arange(-s_H_max,s_H_max+1)        #t for s_H_max

#######################################################################
om = np.vectorize(fn.omega,otypes=[float])
wig_val = np.vectorize(fn.wig,otypes=[float])

#######################################################################
#Creating profile for the radial variation
def B0_st_profile(r):
    beta = 1e-4/r**3  #10G on surface
    #1e5 Gauss at tachocline
    beta += np.exp(-0.5*((r-0.7)/0.01)**2)
    #1e7 Gauss at core
    beta += 100*np.exp(-0.5*(r/0.1)**2)

    #Saving the profile of B_radial_Strength used
    plt.figure()
    plt.semilogy(r,beta)
    plt.savefig('Splitting_functions/B_field_profile.png',dpi=100)
    plt.close()
    return beta

#function to return the B-component of all s,t satisfying realness of B
def ret_Bcomps(s,r):
    B0_st = B0_st_profile(r) * (1 + random.random() * 1.0j)
    #Computing array to impose the solenoidal condition
    gradr_B0st = (np.gradient(r**2*B0_st,r)/r)[np.newaxis,:]/om(s,0)[:,np.newaxis]      #shape s x r

    #To compute a new random field every time
    # rand_factors = np.zeros((len(s),2*smax+1))
    # one_factors = np.zeros((len(s),2*smax+1))
    # for s0 in range(len(s)):
    #     scurrent = s[s0]
    #     #filling in only the positive half of t-axis for all s
    #     rand_factors[s0,smax:smax+scurrent+1] = np.array([random.random() for i in range(scurrent+1)]) #shape s x t
    #     one_factors[s0,smax:smax+scurrent+1] = np.array([1.0 for i in range(scurrent+1)]) #shape s x t
    # np.save('Splitting_functions/rand_factors.npy',rand_factors)
    # np.save('Splitting_functions/one_factors.npy',one_factors)

    #To use the same stored random field
    rand_factors = np.load('Splitting_functions/rand_factors.npy')
    one_factors = np.load('Splitting_functions/one_factors.npy')
    # Saving the profile of random numbers in s x t space
    plt.pcolormesh(rand_factors)
    plt.savefig('Splitting_functions/random_st.png',dpi=100)
    plt.close()

    #Inserting randomness in the various components
    B0_st = B0_st[np.newaxis,:]*rand_factors[:,:,np.newaxis]     #shape s x t x r
    gradr_B0st = gradr_B0st[:,np.newaxis,:]*rand_factors[:,:,np.newaxis]
    Bp_st = rand_factors[:,:,np.newaxis]*gradr_B0st    #shape s x t x r

    #Implementing the solenoidal condition
    Bm_st = gradr_B0st - Bp_st

    #Now we have the positive half of t axis filled for all mu and s satisfying div.B = 0

    #Next we shall fill in the negative half of t-axis

    B0_st[:,smax-1::-1,:] = (-1)**(np.abs(t_smax))[np.newaxis,smax+1:,np.newaxis] * np.conj(B0_st[:,smax+1:,:])
    Bp_st[:,smax-1::-1,:] = (-1)**(np.abs(t_smax))[np.newaxis,smax+1:,np.newaxis] * np.conj(Bm_st[:,smax+1:,:])
    Bm_st[:,smax::-1,:] = (-1)**(np.abs(t_smax))[np.newaxis,smax:,np.newaxis] * np.conj(Bp_st[:,smax:,:])
    #To keep B^-_s0 = B^+_s0*
    Bm_st[:,smax,:] = np.conj(Bp_st[:,smax,:])

    #Keeping the B^0_s0 components real as they have to be
    B0_st[:,smax,:] = np.real(B0_st[:,smax,:])

    B_st_total = np.zeros((3,len(s),2*smax+1,len(r)),dtype='complex128')
    B_st_total[0] = Bm_st
    B_st_total[1] = B0_st
    B_st_total[2] = Bp_st
    return B_st_total

#Check real-ness of B

def isBreal():
    B_st_comp = ret_Bcomps(s,r)
    
    diff_arr = np.zeros((len(s),2*smax+1,len(r)),dtype='complex128')

    diff_arr += B_st_comp[1,:,:,:] - (-1)**(np.abs(t_smax))[np.newaxis,:,np.newaxis]*np.conj(B_st_comp[1,:,2*smax::-1,:])
    diff_arr += B_st_comp[0,:,:,:] - (-1)**(np.abs(t_smax))[np.newaxis,:,np.newaxis]*np.conj(B_st_comp[2,:,2*smax::-1,:])

    print(np.amax(diff_arr.real),np.amax(diff_arr.imag))



#getting the kernel components

m = np.arange(-l,l+1,1)    #-l<=m<=l
m_ = np.arange(-l_,l_+1,1)  #-l_<=m<=l_ 

kern = gkerns.Hkernels(n_,l_,m_,n,l,m,s_H_arr,r,False)
Bmm,B0m,B00,Bpm,Bp0,Bpp = kern.ret_splittingfunction_terms(smoothen=True)

npts = 300   #check the npts in get_kernels
r_new = np.linspace(np.amin(r),np.amax(r),npts)
r = r_new

# B_st = ret_Bcomps2(s,r)     #computes all components of all st of 

#finding the H-components
Hmm_str = np.zeros((len(s_H_arr),2*s_H_max+1,len(r)),dtype='complex128')
H0m_str = np.zeros((len(s_H_arr),2*s_H_max+1,len(r)),dtype='complex128')
H00_str = np.zeros((len(s_H_arr),2*s_H_max+1,len(r)),dtype='complex128')
Hpm_str = np.zeros((len(s_H_arr),2*s_H_max+1,len(r)),dtype='complex128')
H0p_str = np.zeros((len(s_H_arr),2*s_H_max+1,len(r)),dtype='complex128')
Hpp_str = np.zeros((len(s_H_arr),2*s_H_max+1,len(r)),dtype='complex128')

mu = np.array([-1,0,1])
nu = np.array([-1,0,1])


#Getting the B_mu_st field to feed into manufacturing H_munu_st

B_st = ret_Bcomps(s,r)

#manufacturing the H_munu_st 

def ret_Hcomps(s0,t0):
    B_munu_s1t12t2 = B_st[:,np.newaxis,:,:,np.newaxis,np.newaxis,:]*B_st[np.newaxis,:,np.newaxis,np.newaxis,:,:,:]
    t = np.arange(-smax,smax+1)
    mumu,nunu,ss1,tt1,ss2,tt2 = np.meshgrid(mu,nu,s,t,s,t,indexing='ij')
    
    prefactor = (-1)**(np.abs(mumu+nunu+t0)) * np.sqrt((2*s0+1.)*(2*ss1+1.)*(2*ss2+1.)/(4.* np.pi))
    prefactor *= wig_val(s0,ss1,ss2,-t0,tt1,tt2) * wig_val(s0,ss1,ss2,-(mumu+nunu),mumu,nunu)

    H_st_mat = prefactor[:,:,:,:,:,:,np.newaxis] * B_munu_s1t12t2    
    H_st_mat = np.sum(H_st_mat,axis=(2,3,4,5))

    return H_st_mat

#matrix to store H_munu_st_r
H_str_total = np.zeros((3,3,len(s_H_arr),2*s_H_max+1,len(r)),dtype='complex128')

for s0 in range(len(s_H_arr)):     
    scurrent = s_H_arr[s0]
    for t0 in range(-scurrent,scurrent+1):
            Htotal_str = ret_Hcomps(scurrent,t0)
            H_str_total[:,:,s0,t0+s_H_max,:] = Htotal_str

            Hmm_str[s0,t0+s_H_max,:] = H_str_total[0,0,s0,t0+s_H_max,:]
            H0m_str[s0,t0+s_H_max,:] = H_str_total[1,0,s0,t0+s_H_max,:]
            H00_str[s0,t0+s_H_max,:] = H_str_total[1,1,s0,t0+s_H_max,:]
            Hpm_str[s0,t0+s_H_max,:] = H_str_total[2,0,s0,t0+s_H_max,:]
            H0p_str[s0,t0+s_H_max,:] = H_str_total[1,2,s0,t0+s_H_max,:]
            Hpp_str[s0,t0+s_H_max,:] = H_str_total[2,2,s0,t0+s_H_max,:]

    print(scurrent)

#to check the realness of H, which should naturally follow from the realness of B
def isHreal():
    diff_arr = np.zeros((3,3,len(s_H_arr),2*s_H_max+1,len(r)),dtype='complex128')

    diff_arr += H_str_total - (-1)**(np.abs(t_sHmax))[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis]*np.conj(H_str_total[2::-1,2::-1,:,2*s_H_max::-1,:])

    print(np.amax(np.abs(diff_arr.real)),np.amax(np.abs(diff_arr.imag)))

#mesh creation for plotting 
theta = np.linspace(-np.pi/2,np.pi/2,360)
phi = np.linspace(-np.pi,np.pi,720)

phph,thth = np.meshgrid(phi,theta)

# find integrand by summing all component
Lambda_str = Hpp_str*Bpp[:,np.newaxis,:] + H00_str*B00[:,np.newaxis,:] + Hmm_str*Bmm[:,np.newaxis,:] \
        + 2*Hpm_str*Bpm[:,np.newaxis,:] + 2*H0m_str*B0m[:,np.newaxis,:] + 2*H0p_str*Bp0[:,np.newaxis,:]

#radial integral
Lambda_st = scipy.integrate.trapz(Lambda_str*(r**2),x=r,axis=2)

sf = np.zeros(np.shape(thth),dtype='complex128') #variable to store the splitting functions

for s0 in range(0,len(s_H_arr)):
    scurrent = s_H_arr[s0]
    for t0 in range(-scurrent,scurrent+1):
        sf += Lambda_st[s0,t0+s_H_max]*sp.sph_harm(t0,scurrent,phph+np.pi,thth+np.pi/2.0)

sf = sf/(2*omega_0**2)  #normalizing

#Plotting the real part which shall be an even function of phi
figr = plt.figure()
axr = figr.add_subplot(111, projection='aitoff')
#plotting the percentage change in frequency
imr = axr.pcolormesh(phph,thth,np.real(sf)*100, cmap=plt.cm.jet)
figr.colorbar(imr,ax=axr)
plt.savefig('Splitting_functions/SF_real2.png',dpi=400)

#Plotting the imaginary part which shall be an odd function of phi
figi = plt.figure()
axi = figi.add_subplot(111, projection='aitoff')
#plotting the percentage change in frequency
imi = axi.pcolormesh(phph,thth,np.imag(sf)*100, cmap=plt.cm.jet)
figi.colorbar(imi,ax=axi)

