import numpy as np
import scipy.special as sp 
import h_components as hcomps
import get_kernels_herm as gkerns
import functions as fn
import scipy.integrate
import matplotlib.pyplot as plt 
plt.ion()

n,l = 2,4
n_,l_ = 2,4

def uniform_Hprofile(r):
    beta = 1e-4/r**3  #10G on surface
    #1e5 Gauss at tachocline
    beta += np.exp(-0.5*((r-0.7)/0.01)**2)
    #1e7 Gauss at core
    beta += 100*np.exp(-0.5*(r/0.1)**2)
    return beta


r = np.loadtxt('r.dat')
r_start = 0.68
r_end = 1.0
start_ind = fn.nearest_index(r,r_start)
end_ind = fn.nearest_index(r,r_end)
r = r[start_ind:end_ind+1]

B_mag = uniform_Hprofile(r)
H_mag = B_mag**2

#Specifying Lorentz stress component
hfac = np.array([1,1,1,1,1,1])

#distributing the components
hmm = hfac[0]*H_mag
h0m = hfac[1]*H_mag
h00 = hfac[2]*H_mag
hpm = hfac[3]*H_mag
hp0 = hfac[4]*H_mag
hpp = hfac[5]*H_mag

#Lorentz stress components' order
s = np.array([0,1,2,3,4,5,6])
t = 4

#The specific angular degree to plot
s0 = 4

m = np.arange(-l,l+1,1)    #-l<=m<=l
m_ = np.arange(-l_,l_+1,1)  #-l_<=m<=l_ 

kern = gkerns.Hkernels(n_,l_,m_,n,l,m,s,r,False)
Bmm,B0m,B00,Bpm,Bp0,Bpp = kern.ret_kerns()
#sys.exit()

#find integrand by summing all component
Lambda_sr = hpp[np.newaxis,:]*Bpp + h00[np.newaxis,:]*B00 + hmm[np.newaxis,:]*Bmm \
        + 2*hpm[np.newaxis,:]*Bpm + 2*h0m[np.newaxis,:]*B0m + 2*hp0[np.newaxis,:]*Bp0

#summing over s before carrying out radial integral
Lambda_r = Lambda_sr[m[0]+t,m[0],s0,:]

#radial integral
Lambda = scipy.integrate.trapz(Lambda_r*(r**2),x=r,axis=0)



#plotting on Mollweide projection
theta = np.linspace(-np.pi/2,np.pi/2,360)
phi = np.linspace(-np.pi,np.pi,720)

tt,pp = np.meshgrid(theta,phi,indexing='ij')

z = Lambda*sp.sph_harm(s0,t,pp,tt)

fig = plt.figure()
ax = fig.add_subplot(111, projection='mollweide')

im = ax.pcolormesh(pp,tt,np.real(z), cmap=plt.cm.jet)

plt.show()
