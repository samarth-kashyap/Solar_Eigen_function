# python library of functions used in Lorentz stress kernel evaluation

import numpy as np
from sympy.physics.wigner import wigner_3j
from sympy import N as sympy_eval
from scipy.signal import savgol_filter
from scipy import interpolate
import scipy.special as special
import sympy as sy
from math import factorial as fac
import matplotlib.pyplot as plt
import math

P_j = np.array([])

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

def gam(s):
    return np.sqrt((2.*s+1.)/ (4.* np.pi))

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
    gamma_s = np.sqrt((2*s0 + 1)/(4.0*np.pi))
    B_mu_t_r = np.zeros((3,2*s0+1,len(r)),dtype=complex)
    nperf = np.vectorize(math.erf)
    if(field_type=='mixed'):
        R1_ind = np.argmin(np.abs(r-R1))
        R2_ind = np.argmin(np.abs(r-R2))
        b = 0.5*(1+nperf(70*(r-(R1+R2)/2.0)))
        a = b - np.gradient(b,r)*r
    
    beta = lambda r: 1e-4/r**3  #10G on surface

    #1e5 Gauss at tachocline
    alpha = np.exp(-0.5*((r-0.7)/0.01)**2)
    #1e7 Gauss at core
    alpha += 100*np.exp(-0.5*(r/0.1)**2)
    
    if(field_type == 'dipolar'):
        B_mu_t_r[:,s0,:] = omega(s0,0) * 1./np.sqrt(2.) \
                                * np.outer(np.array([1., -2., 1.]),beta(r))
    elif(field_type == 'toroidal'):
            B_mu_t_r[:,s0,:] = omega(s0,0) * 1./np.sqrt(2.) \
                                    * np.outer(np.array([-1j, 0. , 1j]),alpha[:])
    else:
            B_mu_t_r[:,s0,:R1_ind] = omega(s0,0) * 1./np.sqrt(2.) \
                                    * np.outer(np.array([-1j, 0. , 1j]),\
                                            alpha[:R1_ind])
            B_mu_t_r[:,s0,R2_ind:] = omega(s0,0) * 1./np.sqrt(2.) \
                                    * np.outer(np.array([1., -2., 1.]),\
                                            beta(r[R2_ind:]))
            B_mu_t_r[:,s0,R1_ind:R2_ind] = omega(s0,0) * 1./np.sqrt(2.) \
                                    * np.array([1., -2., 1.])[:,np.newaxis]*\
                                            beta(r[R1_ind:R2_ind])*np.array([a[R1_ind:R2_ind],\
                                            b[R1_ind:R2_ind],a[R1_ind:R2_ind]])
            B_mu_t_r[:,s0,R1_ind:R2_ind] += omega(s0,0) * 1./np.sqrt(2.) \
                                    * np.outer(np.array([-1j, 0., 1j]),\
                                            alpha[R1_ind:R2_ind])
                                            
    return B_mu_t_r/gamma_s
	
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
        print('infinity encountered in P_lmN evaluation. result not reliable')
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
    
def P_a(l,i):
    global P_j

    P_l_vec = np.vectorize(special.legendre(i))
    L = np.sqrt(l*(l+1))
    m = np.arange(-l,l+1,1)

    #returns 2*l+1 values of P_j^l(m)
    P = np.zeros(2*l+1)
    
    if (l == 0):
        print("l can't be zero in discretised Legendre P")
        return None
    
    #P_0^l(m) = l
    if(i==0): 
        P += l
    
    #creating P''(m) for all m's belonging to l. Needed for c_ij
    P_pp_i = L*P_l_vec((1.*m)/L)
    P_p_i = np.zeros(2*l+1)

    for j in range(0,i,1):
        c_ij_num = 0.0
        c_ij_denom = 0.0 
        P_j_l = P_j[j]  #using pre-computed P_j^l(m)'s
        c_ij_num = np.sum(P_pp_i*P_j_l)
        c_ij_denom = np.sum(P_j_l**2)
        c_ij = c_ij_num/c_ij_denom
        P_p_i -= c_ij*P_j_l
    
    P_p_i += P_pp_i

    #returns an array of length (2*l+1) 
    P = l*P_p_i/P_p_i[-1]

    if(i==0):
        P_j = np.append(P_j,P)
        P_j = np.reshape(P_j,(1,2*l+1))
    else: P_j = np.append(P_j,np.reshape(P,(1,2*l+1)),axis=0)
    return P
    
def a_coeff(del_om, l, jmax, plot_switch = False):
    """a[0] is actually a_1"""

    #this part is just for plotting the basis P's
    if(plot_switch):
        for j in range(jmax+1): P_a(l,j)

        m = np.arange(-l,l+1,1)
        for i in range(jmax+1):
            plt.plot(m,P_j[i],label='%i'%i)
        plt.legend()
        plt.ylabel('$\mathcal{P}_{j}^{(%i)}$'%l)
        plt.xlabel('m')
        plt.show()
        return 0

    #this is where the a-coeffs are computed
    a = np.zeros(jmax+1)
    for j in range(jmax+1):
        for m in np.arange(-l,l+1,1):
            a[j] += del_om[m+l] * P_a(l,j)
        a[j] *= (j+0.5) / l**3

    return a

def a_coeff_matinv(del_om, l, jmax):
    """Inverting for a-coeff from matrix inversion. AC = B. A contains coeffs"""

    P_a_vec = np.vectorize(P_a)

    A_i = np.zeros(jmax+1)
    j = np.arange(0,jmax+1,1)
    m = np.arange(-l,l+1,1)
    jj,mm = np.meshgrid(j,m,indexing='ij')

    P_j_m = P_a_vec(mm,l,jj)
    C_j_i = np.matmul(P_j_m,np.transpose(P_j_m))
    B_j = np.matmul(P_j_m,del_om)

    A_i = np.linalg.solve(C_j_i,B_j)
    return A_i

#finding a-coefficients using GSO
def a_coeff_GSO(del_om,l,jmax):
    """a[0] is actually a_1"""
    global P_j

    a = np.ones(jmax+1)
    a_num = np.zeros(jmax+1)
    a_denom = np.zeros(jmax+1)
    for j in range(jmax+1): 
        P_l_j =  P_a(l,j)
        a_num[j] += np.sum(del_om * P_l_j)
        a_denom[j] += np.sum(P_l_j**2)
    a = a_num/a_denom

    P_j = np.array([])

    return a

def find_omega(n,l):
    return np.loadtxt('muhz.dat')[find_nl(n,l)] * 1e-6 /np.loadtxt('OM.dat')    

def plot_freqs(f_dpt,f_qdpt,nl_list,case,saveCond=False,f_DR=np.array([])):
    OM = np.loadtxt('OM.dat')
    omega_list = np.loadtxt('muhz.dat')  #normlaised frequency list
    omega_nl = np.array([omega_list[find_nl(mode[0], mode[1])] for mode in nl_list])

    dpi = 80
    plt.figure(num=None, figsize=(8, 6), dpi=dpi, facecolor='w', edgecolor='k')
    plt.subplot(2,1,1)

    l_local = 2*nl_list[0,1]+1
    m_local = np.arange(0,l_local)
    plt.plot(m_local,np.ones(len(m_local))*omega_nl[0],'g--',label='Unperturbed')
    for i in range(1,len(omega_nl)):
        m_local = np.arange(l_local,l_local+2*nl_list[i,1]+1)
        plt.plot(m_local,np.ones(len(m_local))*omega_nl[i],'g--')
        l_local += 2*nl_list[i,1]+1

    plt.plot(f_dpt,label='Degenerate')
    plt.plot(f_qdpt,label='Quasi-Degenerate')
    plt.legend()

    f_dpt_min = np.amin(f_dpt)
    f_dpt_max = np.amax(f_dpt)
    freq_arr = np.arange(f_dpt_min,f_dpt_max,(f_dpt_max-f_dpt_min)/100)

    l_local = 0
    title_str = ''
    for i in range(len(nl_list)-1):
        l_local += (2*nl_list[i,1]+1)
        plt.plot(l_local*np.ones(len(freq_arr)),freq_arr,'--k',alpha = 0.3)
        title_str = title_str + 'n,l = ' + str(nl_list[i,0]) + ',' + str(nl_list[i,1]) + ';'

    title_str = title_str + 'n,l = ' + str(nl_list[-1,0]) + ',' + str(nl_list[-1,1]) + ';'

    plt.title(title_str)
    plt.ylabel('Frequency in $\mu$Hz',fontsize=14)
    plt.xlabel('Cumulative m',fontsize=12)

    plt.subplot(2,1,2)

    if(len(f_DR)>0):
        erf_dpt_min = np.amin(f_qdpt-f_DR)
        erf_dpt_max = np.amax(f_qdpt-f_DR)
    else:
        erf_dpt_min = np.amin(f_dpt-f_qdpt)
        erf_dpt_max = np.amax(f_dpt-f_qdpt)
    erfreq_arr = np.arange(erf_dpt_min,erf_dpt_max,(erf_dpt_max-erf_dpt_min)/100)

    l_local = 0
    for i in range(len(nl_list)-1):
        l_local += (2*nl_list[i,1]+1)
        plt.plot(l_local*np.ones(len(erfreq_arr)),erfreq_arr,'--k',alpha = 0.3)

    if(len(f_DR)>0): 
        plt.plot(f_qdpt-f_DR)
        plt.ylabel('$f_{QDM} - f_{DR}$ in $\mu$Hz',fontsize=14)
    else: 
        plt.plot(f_dpt-f_qdpt)
        plt.ylabel('$f_D - f_{QD}$ in $\mu$Hz',fontsize=14)
    plt.xlabel('Cumulative m',fontsize=12)
    plt.show()

    fname = ''
    for i in range(len(nl_list)):
        fname = fname + str(nl_list[i,0]) + '_' + str(nl_list[i,1])
        if(i!=len(nl_list)-1): fname = fname + '-'
        else: fname = fname + case

    if(saveCond == True):
        plt.savefig('./figures/'+fname+'.eps',dpi=dpi)

