# python library of functions used in Lorentz stress kernel evaluation

import numpy as np
from sympy.physics.wigner import wigner_3j
from sympy import N as sympy_eval
from scipy.signal import savgol_filter
from scipy import interpolate
import sympy as sy
from math import factorial as fac
import math
#evaluation
def wig(l1,l2,l3,m1,m2,m3):
	"""returns numerical value of wigner3j symbol"""
	if (np.abs(m1) > l1 or np.abs(m2) > l2 or np.abs(m3) > l3):
	    return 0.
	return(sympy_eval(wigner_3j(l1,l2,l3,m1,m2,m3)))

def omega(l,n):
	"""returns numerical value of \Omega_l^n"""
	if (np.abs(n) > l):	
		return 0.
	return np.sqrt(0.5*(l+n)*(l-n+1.))

def deriv(y,x):
	"""returns derivative of y wrt x. same len as x and y"""
	if(len(x) != len(y)):	
		print("lengths dont match")
		exit()
	dy = np.empty(y.shape)
	dy[0] = (y[1]-y[0]) / (x[1] - x[0])
	dy[-1] = (dy[-1] - dy[-2]) / (x[-1] - x[-2])
	dy[1:-1] = (y[2:] - y[:-2]) / (x[2:]-x[:-2])
	return dy

def deriv2(y,x):
	"""returns second derivative of y wrt x"""
	if(len(x) != len(y)):	
		print("lengths dont match")
		exit()
	l = len(y)	
	ret = np.zeros(l)
	for i in range(1,l-1):
		xf,yf = x[i+1], y[i+1]
		xb,yb = x[i-1], y[i-1]
		xx,yy = x[i], y[i]
		ret[i] = 2./(xf-xb) * ((yf-yy)/(xf-xx) - (yy-yb)/(xx-xb))
	ret[0], ret[-1] = ret[1], ret[-2]
	return ret
	
def nearest_index(array, value):
	"""returns index of object nearest to value in array"""
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx

#loading
nl_list = np.loadtxt('nl.dat')	
def find_nl(n,l):
	"""returns nl index from given n and l"""
	for i in range(len(nl_list)):
		if (np.array_equal(nl_list[i],np.array([n,l]))):
			return i
	return None #when mode not found in nl_lsit

def find_mode(nl):
	"""returns (n,l) for given nl"""
	return int(nl_list[nl][0]), int(nl_list[nl][1])

def load_eig(n,l,eig_dir):
	"""returns U,V for mode n,l stored in directory eig_dir"""
	nl = find_nl(n,l)
	if (nl == None):
		print("mode doesn't exist in nl_list. exiting.")
		exit()
	U = np.loadtxt(eig_dir + '/'+'U'+str(nl)+'.dat')
	V = np.loadtxt(eig_dir + '/'+'V'+str(nl)+'.dat')	
	return U,V
	
def smooth(U,r,window,order,npts):

	#creating interpolated function
	U_interp = interpolate.interp1d(r,U)
	#creating new grid
	r_new = np.linspace(np.amin(r),np.amax(r),npts)

	#smoothening the U
	U_sm = savgol_filter(U_interp(r_new), window, order)

	#taking derivative on smoothened U
	dU = np.gradient(U_sm,r_new)
	#smoothening the derivative obtained from smoothened U
	dU_sm = savgol_filter(dU, window, order)

	#obtaining the second derivative
	ddU = np.gradient(dU_sm,r_new)
	ddU_sm = savgol_filter(ddU, window, order)

	return U_sm, dU_sm, ddU_sm

def kron_delta(i,j):
    if (i==j):
        return 1.
    else:
        return 0.
        
def getB_comps(s0,r,R1,R2,field_type):
    """function to get the components of B_field"""    
    
    B_mu_t_r = np.zeros((3,2*s0+1,len(r)),dtype=complex)
    nperf = np.vectorize(math.erf)
    if(field_type=='mixed'):
        R1_ind = np.argmin(np.abs(r-R1))
        R2_ind = np.argmin(np.abs(r-R2))
        b = 0.5*(1+nperf(70*(r-(R1+R2)/2.0)))
        a = b - np.gradient(b)*r
    
    beta = lambda r: 1./r**3

    alpha = beta  #for now keeping the radial dependence same as dipolar
    
    if(field_type == 'dipolar'):
        B_mu_t_r[:,s0,:] = 1e-4 * omega(s0,0) * 1./np.sqrt(2.) \
                                * np.outer(np.array([1., -2., 1.]),beta(r))
    elif(field_type == 'toroidal'):
            B_mu_t_r[:,s0,:] = 1e-4 * omega(s0,0) * 1./np.sqrt(2.) \
                                    * np.outer(np.array([-1j, 0. , 1j]),alpha(r))
    else:
            B_mu_t_r[:,s0,:R1_ind] = 1e-4 * omega(s0,0) * 1./np.sqrt(2.) \
                                    * np.outer(np.array([-1j, 0. , 1j]),\
                                            alpha(r[:R1_ind]))
            B_mu_t_r[:,s0,R2_ind:] = 1e-4 * omega(s0,0) * 1./np.sqrt(2.) \
                                    * np.outer(np.array([1., -2., 1.]),\
                                            beta(r[R2_ind:]))
            B_mu_t_r[:,s0,R1_ind:R2_ind] = 1e-4 * omega(s0,0) * 1./np.sqrt(2.) \
                                    * np.array([1., -2., 1.])[:,np.newaxis]*\
                                            beta(r[R1_ind:R2_ind])*np.array([a[R1_ind:R2_ind],\
                                            b[R1_ind:R2_ind],a[R1_ind:R2_ind]])
            B_mu_t_r[:,s0,R1_ind:R2_ind] += 1e-4 * omega(s0,0) * 1./np.sqrt(2.) \
                                    * np.outer(np.array([-1j, 0., 1j]),\
                                            alpha(r[R1_ind:R2_ind]))
                                            
    return B_mu_t_r
	
def P(mu,l,m,N):
    """generalised associated legendre function"""
    x = sy.Symbol('x')
    ret = sy.simplify(sy.diff((x-1)**(l-N) * (x+1)**(l+N), x, l-m))
    if (type(mu) == np.ndarray):
        temp = np.ndarray.flatten(mu)
        temp = np.array([ret.evalf(subs={x:t}) for t in temp])
        ret = np.reshape(temp, mu.shape)
    else:    
        ret = ret.evalf(subs={x:mu})
    ret *= 1./2**l * 1./np.sqrt(fac(l+N)*fac(l-N)) * np.sqrt(1.*fac(l+m) / fac(l-m))
    ret /= np.sqrt((1.-mu)**(m-N) * (1.+mu)**(m+N)) 
    if np.any(ret == np.inf):
        print 'infinity encountered in P_lmN evaluation. result not reliable'
    return ret

def d_rotate(beta,l,m_,m):
    """spherical harmonic rotation matrix element m,m_"""    
    if(beta == 0):
        if (m==m_): 
            return 1
        else:
            return 0
    return  P(np.cos(beta*np.pi/180.),l,m,m_)
    
def d_rotate_matrix(beta,l):
    """returns spherical harmonic rotation matrix"""
    ret = np.empty((2*l+1,2*l+1))
    for i in range(2*l+1):
        for j in range(2*l+1):
            ret[i,j] = d_rotate(beta,l,i-l,j-l)
    return ret

def d_rotate_matrix_padded(beta,l,l_large):
    """returns d_rotate matrix padded with 0s in larger 2l_large+1 X 2l_large+1 matrix"""
    ret = np.zeros((2*l_large+1,2*l_large+1))
    ret[(l_large-l):l_large+l+1,(l_large-l):l_large+l+1] = d_rotate_matrix(beta,l)
    return ret
    
def Y_lmN(theta,phi,l,m,N):
    ret = np.sqrt((2.*l+1)/(4.*np.pi)) * P(np.cos(theta),l,m,N) * np.exp(1j*m*phi)
    return ret
