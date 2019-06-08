import numpy as np
import functions as fn

class getHcomps:
    """Class to compute the H-coefficients for Lorentz stresses"""

    def __init__(self,s,m,l_b,m_b,r,b_r):
        self.mu = np.array([-1,0,1])
        self.nu = np.array([-1,0,1])
        self.s = s 
        self.m = m
        self.l_b = l_b
        self.m_b = m_b
        self.r = r
        self.b_r = b_r

    def ret_hcomps(self):
        
        t = np.arange(-np.max(np.abs(self.s)),np.max(np.abs(self.s))+1,1)
        mumu,nunu,ss,tt = np.meshgrid(self.mu,self.nu,self.s,t,indexing='ij')

        wig_calc = np.vectorize(fn.wig)


        b_mu_nu = np.array([-1j*fn.omega(self.l_b,0),0,1j*fn.omega(self.l_b,0)])


        B_mu_nu = np.real(np.outer(b_mu_nu,b_mu_nu))
        B_mu_nu_r = B_mu_nu[:,:,np.newaxis]*self.b_r[np.newaxis,:] 
    

        #signs to be checked
        wig1 = wig_calc(self.l_b,ss,self.l_b,mumu,-(mumu+nunu),nunu)
        wig2 = wig_calc(self.l_b,ss,self.l_b,self.m_b,-tt,self.m_b)
        #factor of 9 needs to be replaced with generic l expression
        H = ((-1)**(np.abs(tt+mumu+nunu))) \
                *np.sqrt((2*self.l_b+1)*(2*self.l_b+1)*(2*ss+1)/(4*np.pi))*wig1*wig2

        HH = H[:,:,:,:,np.newaxis] \
                *(B_mu_nu_r*self.r[np.newaxis,:])[:,:,np.newaxis,np.newaxis,:]


        HH = HH.astype('float64')

        H_super = np.zeros((len(self.m),len(self.m),len(self.mu), \
                    len(self.nu),len(self.s),len(self.r)))

        mm,mm_ = np.meshgrid(self.m,self.m,indexing='ij')

        for i in t:
            H_super[mm-mm_ == i] = HH[:,:,:,i+np.max(np.abs(self.s)),:] 
            
#        print(np.shape(H_super))
            
        H_super = np.swapaxes(H_super,0,2)
        H_super = np.swapaxes(H_super,1,3)
        
#        print(np.shape(H_super))
                
        return H_super


