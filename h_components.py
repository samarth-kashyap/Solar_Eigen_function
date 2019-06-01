import numpy as np
import functions as fn

class getHcomps:
    """Class to compute the H-coefficients for Lorentz stresses"""

    def __init__(self,mu,nu,s,m,l,r):
        self.mu = mu
        self.nu = nu
        self.s = s 
        self.m = m
        self.l = l
        self.r = r

    def ret_hcomps(self):
        
        t = np.arange(-np.max(np.abs(self.s)),np.max(np.abs(self.s))+1,1)
        mumu,nunu,ss,tt = np.meshgrid(self.mu,self.nu,self.s,t,indexing='ij')

        wig_calc = np.vectorize(fn.wig)

        #signs to be checked
        wig1 = wig_calc(1,ss,1,mumu,-(mumu+nunu),nunu)
        wig2 = wig_calc(1,ss,1,0,tt,0)
        #factor of 9 needs to be replaced with generic l expression
        H = np.sqrt((9/(4*np.pi))*(2*ss+1))*wig1*wig2

        HH = H[:,:,:,:,np.newaxis]*self.r[np.newaxis,:]

        HH = HH.astype('float64')

        H_super = np.zeros((len(self.m),len(self.m),len(self.mu), \
                    len(self.nu),len(self.s),len(self.r)))

        mm,mm_ = np.meshgrid(self.m,self.m,indexing='ij')

        for i in t:
            H_super[mm-mm_ == i] = HH[:,:,:,i+np.max(np.abs(self.s)),:] 
                
        return H_super


