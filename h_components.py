import numpy as np
import functions as fn

class getHcomps:
    """Class to compute the H-coefficients for Lorentz stresses"""

    def __init__(self,s,m,s0,t0,r,B_mu_t_r, beta):
        self.mu = np.array([-1,0,1])
        self.nu = np.array([-1,0,1])
        self.s = s
        self.m = m
        self.s0 = s0
        self.t0 = t0
        self.r = r
        self.B_mu_t_r = B_mu_t_r
        self.beta = beta

    def ret_hcomps(self):
        
        t = np.arange(-np.max(np.abs(self.s)),np.max(np.abs(self.s))+1,1)
        mumu,nunu,ss,tt,tt0 = np.meshgrid(self.mu,self.nu,self.s,t,self.t0,indexing='ij')
        
        d_rot = np.vectorize(fn.d_rotate)
        d_matrix = np.zeros((2*self.s0+1, 2*self.s0+1))
        for i in range(2*self.s0+1):
            for j in range(2*self.s0+1):
                d_matrix[i,j] = fn.d_rotate(self.beta, self.s0, i-self.s0, j-self.s0)
        
        self.B_mu_t_r = d_matrix[np.newaxis,:,:,np.newaxis] * self.B_mu_t_r[:,:,np.newaxis,:]
        self.B_mu_t_r = np.sum(self.B_mu_t_r, axis = 1)
        
        
        
#        print self.B_mu_t_r.shape
#        exit()
#        print d_matrix
#        exit()
        
        wig_calc = np.vectorize(fn.wig)
        BB_mu_nu_t_t0_r = np.zeros((3,3,len(t),len(self.t0),len(self.r)),dtype = complex)
        for t_iter in range(-np.max(np.abs(self.s)),np.max(np.abs(self.s))+1):
            for t0_iter in range(-self.s0, self.s0+1):
                if (t0_iter >= max(-self.s0,-self.s0+t_iter) and t0_iter <= min(self.s0,self.s0+t_iter)):
                    BB_mu_nu_t_t0_r[:,:,t_iter,t0_iter,:] = self.B_mu_t_r[:,np.newaxis,t0_iter,:] * self.B_mu_t_r[np.newaxis,:,t_iter-t0_iter,:]
#        BB_mu_nu_t_r = np.sum(BB_mu_nu_t_t0_r, axis=3)
                
#        print BB_mu_nu_t_r.shape
#        print BB_mu_nu_t_r[0,0,:,100]
#        exit()


        #signs to be checked
        wig1 = wig_calc(self.s0,ss,self.s0,mumu,-(mumu+nunu),nunu)
        wig2 = wig_calc(self.s0,ss,self.s0,tt0,-tt,tt-tt0)
        #factor of 9 needs to be replaced with generic l expression
        H = ((-1)**(np.abs(tt+mumu+nunu))) \
                *np.sqrt((2*self.s0+1)*(2*self.s0+1)*(2*ss+1)/(4*np.pi))*wig1*wig2

        HH = H[:,:,:,:,:,np.newaxis] *BB_mu_nu_t_t0_r[:,:,np.newaxis,:,:,:]
        HH = HH.astype('complex128')
        HH = np.sum(HH, axis=4) #summing over t0
        
#        print HH.shape
#        print HH[0,0,-1,:,100]

        H_super = np.zeros((len(self.m),len(self.m),len(self.mu), \
                    len(self.nu),len(self.s),len(self.r)),dtype = complex)

        mm,mm_ = np.meshgrid(self.m,self.m,indexing='ij')

        for i in t:
            H_super[mm_-mm == i] = HH[:,:,:,i+np.max(np.abs(self.s)),:]
            
#        print(np.shape(H_super))
            
        H_super = np.swapaxes(H_super,0,2)
        H_super = np.swapaxes(H_super,1,3)
        
#        print(np.shape(H_super))
                
        return H_super


