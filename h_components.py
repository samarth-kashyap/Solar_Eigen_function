import numpy as np
import functions as fn
class getHcomps:
    """Class to compute the H-coefficients for Lorentz stresses"""

    def __init__(self,s,m_,m,s0,t0,r,B_mu_t_r, beta = 0.):
        self.mu = np.array([-1,0,1])
        self.nu = np.array([-1,0,1])
        self.s = s
        self.m = m
        self.m_ = m_
        self.s0 = s0
        self.t0 = t0
        self.r = r
        self.B_mu_t_r = B_mu_t_r #may or may not have t dimension
        self.beta = beta

    def ret_hcomps(self):
        s_max = np.max(np.abs(self.s))
        t = np.arange(-s_max,s_max+1,1)
        mumu,nunu,ss,tt,tt0 = np.meshgrid(self.mu,self.nu,self.s,t,self.t0,indexing='ij')

        #tilting B
#        d_matrix = fn.d_rotate_matrix(self.beta,self.s0)        
#        self.B_mu_t_r = d_matrix[np.newaxis,:,:,np.newaxis] * self.B_mu_t_r[:,np.newaxis,:,:]
#        self.B_mu_t_r = np.sum(self.B_mu_t_r, axis = 2)
        
#########TILT TESTING        
#        eps = 1e-4      
#        theta = np.linspace(eps,np.pi-eps,20)
#        phi = np.linspace(0,2*np.pi,21)
#        NN, tt, thth, phph = np.meshgrid(self.mu, self.t0, theta,phi, indexing = 'ij')
#        Yv = np.vectorize(fn.Y_lmN)
#        print 'Yv starts'
#        YY = Yv(thth, phph, self.s0, tt, NN)
#        print 'Yv ends'        
#        B_disp = self.B_mu_t_r[:,:,:, np.newaxis,np.newaxis] * YY[:,:,np.newaxis,:,:]
#        B_disp = np.sum(B_disp, axis = 1)
##        B_disp = np.sum(B_disp, axis = 0)
#        return B_disp[1]        
#########TILT TESTING        
        
        wig_calc = np.vectorize(fn.wig)
        BB_mu_nu_t_t0_r = np.zeros((3,3,len(t),len(self.t0),len(self.r)),dtype = complex)
        for t_iter in range(-np.max(np.abs(self.s)),np.max(np.abs(self.s))+1):
            for t0_iter in range(-self.s0, self.s0+1):
                if (t0_iter >= max(-self.s0,-self.s0+t_iter) and t0_iter <= min(self.s0,self.s0+t_iter)):
                    BB_mu_nu_t_t0_r[:,:,t_iter,t0_iter,:] = self.B_mu_t_r[:,np.newaxis,t0_iter,:] * self.B_mu_t_r[np.newaxis,:,t_iter-t0_iter,:]

        #signs to be checked
        wig1 = wig_calc(self.s0,ss,self.s0,mumu,-(mumu+nunu),nunu)
        wig2 = wig_calc(self.s0,ss,self.s0,tt0,-tt,tt-tt0)
        #factor of 9 needs to be replaced with generic l expression
        H = ((-1)**(np.abs(tt+mumu+nunu))) \
                *np.sqrt((2*self.s0+1)*(2*self.s0+1)*(2*ss+1)/(4*np.pi))*wig1*wig2

        HH = H[:,:,:,:,:,np.newaxis] *BB_mu_nu_t_t0_r[:,:,np.newaxis,:,:,:]
        HH = HH.astype('complex128')
        HH = np.sum(HH, axis=4) #summing over t0

        # #tilting HH     
        # for s_iter in range(s_max+1):
        #     d = fn.d_rotate_matrix_padded(self.beta,s_iter,s_max)
        #     temp = HH[:,:,s_iter,np.newaxis,:,:] * d[np.newaxis,np.newaxis,:,:,np.newaxis]
        #     HH[:,:,s_iter,:,:] = np.sum(temp ,axis = 3)

        
        H_super = np.zeros((len(self.m_),len(self.m),len(self.mu), \
                    len(self.nu),len(self.s),len(self.r)),dtype = complex)

        mm_,mm = np.meshgrid(self.m_,self.m,indexing='ij')

        for i in t:
            H_super[mm_-mm == i] = HH[:,:,:,i+np.max(np.abs(self.s)),:]
            
            
        H_super = np.swapaxes(H_super,0,2)
        H_super = np.swapaxes(H_super,1,3)
                
        return H_super
        
    def ret_hcomps_axis_symm(self):
        s_max = np.max(np.abs(self.s)) #remove odd s components
        mumu,nunu,ss = np.meshgrid(self.mu,self.nu,self.s,indexing='ij')     
        
        wig_calc = np.vectorize(fn.wig)
        B_mu_r = self.B_mu_t_r
        BB_mu_nu_r = B_mu_r[:,np.newaxis,:] * B_mu_r[np.newaxis,:,:]

        #signs to be checked
        wig1 = wig_calc(self.s0,ss,self.s0,mumu,-(mumu+nunu),nunu)
        wig2 = wig_calc(self.s0,ss,self.s0,0,0,0) #can cut down on odd s calculation
        H = ((-1)**(np.abs(mumu+nunu))) \
                *np.sqrt((2*self.s0+1)*(2*self.s0+1)*(2*ss+1)/(4*np.pi))*wig1*wig2

        HH = H[:,:,:,np.newaxis] *BB_mu_nu_r[:,:,np.newaxis,:]
        HH = HH.astype('complex128')       
                
        return HH
    

        
