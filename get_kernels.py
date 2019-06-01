import numpy as np
import matplotlib.pyplot as plt
import functions as fn
from os import getcwd
import timing
clock1 = timing.stopclock()
tstamp = clock1.lap


class Hkernels:
    """This class handles l parameters of the kernel"""
    #setting up shorthand repeatedly used in kernel evaluation

    def __init__(self,l,l_,r,rpts,ss1=0,ss2=0):
        self.l = l
        self.l_ = l_
        # self.ss = ss1
        # self.s = ss2[0,0,:,:]
        self.ss = 0
        self.s = 0
        self.rpts = rpts
        self.r = r

    def isol_multiplet(self,n,l,s):
        n_ = n
        l_ = l
        m = np.arange(-l,l+1,1)
        m_ = np.arange(-l,l+1,1)

        mm,mm_,ss1 = np.meshgrid(m,m_,s,indexing='ij')
        __,__,ss2,__ = np.meshgrid(m,m,s,self.r,indexing='ij')

        self.ss = ss1
        self.s = ss2[0,0,:,:]

        Bmm,B0m,B00,Bpm,Bpp,B0p = self.ret_kerns(l,self.ss,l_,mm,mm_,n,n_)
        return Bmm,B0m,B00,Bpm,Bpp,B0p

    def wig_red_o(self,m1,m2,m3):
        '''3j symbol with upper row fixed'''
        wig_vect = np.vectorize(fn.wig)
        return wig_vect(self.l_,self.ss,self.l,m1,m2,m3)

    def wig_red(self,m1,m2,m3):
        '''3j symbol with upper row fixed'''
        wig_vect = np.vectorize(fn.wig)
        return wig_vect(self.l_,self.s,self.l,m1,m2,m3)

    def ret_kerns(self,l,s,l_,m,m_,n,n_):

        nl = fn.find_nl(n,l)
        nl_ = fn.find_nl(n_,l_)

        len_m, len_m_, len_s = np.shape(s)

        #Savitsky golay filter for smoothening
        window = 45  #must be odd
        order = 3

        if(nl == None or nl_ == None):
            print("Mode not found. Exiting."); exit()

        #loading required functions
        eig_dir = (getcwd() + '/eig_files')
        Ui,Vi = fn.load_eig(n,l,eig_dir)
        Ui_,Vi_= fn.load_eig(n_,l_,eig_dir)
        r = np.loadtxt('r.dat')
        rho = np.loadtxt('rho.dat')

        #slicing the radial function acoording to radial grids
        r = r[-self.rpts:]
        rho = rho[-self.rpts:]
        Ui = Ui[-self.rpts:]
        Vi = Vi[-self.rpts:]
        Ui_ = Ui_[-self.rpts:]
        Vi_ = Vi_[-self.rpts:]

        tstamp()
        om = np.vectorize(fn.omega)
        parity_fac = (-1)**(l+l_+s) #parity of selected modes
        prefac = np.sqrt((2*l_+1.) * (2*s+1.) * (2*l+1.) \
                    / (4.* np.pi)) * self.wig_red_o(-m_,m_-m,m)
        tstamp('Computed prefac in')

        #EIGENFUCNTION DERIVATIVES

        #smoothing

        #interpolation params
        #npts = 30000
        #r_new = np.linspace(np.amin(r),np.amax(r),npts)


        #Ui,dUi,d2Ui = fn.smooth(U,r,window,order,npts)
        #Vi,dVi,d2Vi = fn.smooth(V,r,window,order,npts)

        #Ui_,dUi_,d2Ui_ = fn.smooth(U_,r,window,order,npts)
        #Vi_,dVi_,d2Vi_ = fn.smooth(V_,r,window,order,npts)

        #rho_sm, __, __ = fn.smooth(rho,r,window,order,npts)
        ##re-assigning with smoothened variables
        #r = r_new
        #rho = rho_sm

        ##no smoothing
        dUi, dVi = np.gradient(Ui,r), np.gradient(Vi,r)
        dUi_, dVi_ = np.gradient(Ui_,r), np.gradient(Vi_,r)
        d2Ui_,d2Vi_ = np.gradient(dUi_,r), np.gradient(dVi_,r)
        tstamp('Loaded eigfiles in')

        ##making U,U_,V,V_,dU,dU_,dV,dV_,d2U,d2U_,d2V,d2V_ of same shape

        U = np.tile(Ui,(len_s,1))
        V = np.tile(Vi,(len_s,1))
        dU = np.tile(dUi,(len_s,1))
        dV = np.tile(dVi,(len_s,1))
        tstamp('Tiling initial functions in')


        # U_ = np.tile(Ui_,(len_s,len_m,len_m_,1))
        # V_ = np.tile(Vi_,(len_s,len_m,len_m_,1))
        # dU_ = np.tile(dUi_,(len_s,len_m,len_m_,1))
        # dV_ = np.tile(dVi_,(len_s,len_m,len_m_,1))
        # d2U_ = np.tile(d2Ui_,(len_s,len_m,len_m_,1))
        # d2V_ = np.tile(d2Vi_,(len_s,len_m,len_m_,1))

        # r = np.tile(r,(len_s,len_m,len_m_,1))

        U_ = np.tile(Ui_,(len_s,1))
        V_ = np.tile(Vi_,(len_s,1))
        dU_ = np.tile(dUi_,(len_s,1))
        dV_ = np.tile(dVi_,(len_s,1))
        d2U_ = np.tile(d2Ui_,(len_s,1))
        d2V_ = np.tile(d2Vi_,(len_s,1))
        tstamp('Derivatives tiled in')
        r = np.tile(r,(len_s,1))
        rf = np.tile(r,(len_m,len_m_,1,1))
        
        tstamp()

        #B-- EXPRESSION
        Bmm = -r*(self.wig_red(0,-2,2)*om(l,0)*om(l,2)*V*dU_ + self.wig_red(2,-2,0)*om(l_,0)* \
                om(l_,2)*V_*dU)
        Bmm += self.wig_red(1,-2,1)*om(l_,0)*om(l,0)*(U-V)*(U_ - V_ + r*dV_)
        tstamp('Computed Bmm without prefac in')


        # Bmm = np.tile(Bmm,(len_m,len_m_,1,1))


        # Bmm = (((-1)**np.abs(m_))*prefac/(rf**2))[:,:,:,np.newaxis] \
                #  * Bmm[np.newaxis,:,:]
        print('Constructed new axes')
        Bmm = (((-1)**np.abs(m_))*prefac)[:,:,:,np.newaxis] \
                 * (Bmm/r**2)[np.newaxis,:,:]
        tstamp('Bmm corrected in')		
        #B-- EXTRA
        Bmm_ = om(l_,0)*(self.wig_red(2,-2,0)*om(l_,2)*U*(V_ - r*dV_) + om(l,0)*V \
                *(self.wig_red(3,-2,-1)*om(l_,2)*om(l_,3)*V_ + self.wig_red(1,-2,1) \
                *(-U_ + V_ + om(l_,2)**2 *V_ - r*dV_)))

        # Bmm_ = np.tile(Bmm_,(len_m,len_m_,1,1))

        # Bmm_ = (((-1)**np.abs(m_))*prefac/(rf**2))[:,:,:,np.newaxis] \
                #  * Bmm_[np.newaxis,:,:]

        Bmm_ = (((-1)**np.abs(1+m_))*prefac)[:,:,:,np.newaxis] \
                 * (Bmm_/r**2)[np.newaxis,:,:]

        print('Bmm done')


        #B0- EXPRESSION
        B0m = self.wig_red(1,-1,0)*om(l_,0)*(U - (om(l,0)**2)*V)*(U_ - V_ + r*dV_)
        B0m += om(l,0)*(om(l_,0)*(self.wig_red(-1,-1,2)*om(l,2)*V*(U_ - V_ + r*dV_) \
            + 2*r*self.wig_red(2,-1,-1)*om(l_,2)*V_*dV) + self.wig_red(0,-1,1) \
            *((U-V)*(2*U_ - 2*(om(l_,0)**2)*V_ - r*dU_) + r**2 * dU_*dV))

        # B0m = np.tile(B0m,(len_m,len_m_,1,1))
        
        B0m = (0.5*((-1)**np.abs(m_))*prefac)[:,:,:,np.newaxis] \
                * (B0m/r**2)[np.newaxis,:,:]
                
        #B0- EXTRA
        B0m_ = om(l,0)*V*(self.wig_red(2,-1,-1)*om(l_,0)*om(l_,2)*(U_ - 3*V_ + r*dV_) \
                + self.wig_red(0,-1,1)*((2+om(l_,0)**2)*U_ - 2*r*dU_ + om(l_,0)**2 \
                *(-3*V_ + r*dV_)))
        B0m_ += self.wig_red(1,-1,0)*om(l_,0)*U*(U_ - V_ - r*(dU_ - dV_ + r*d2V_))
        

        # B0m_ = np.tile(B0m_,(len_m,len_m_,1,1))

        B0m_ = (0.5*((-1)**np.abs(m_))*prefac)[:,:,:,np.newaxis] \
                * (B0m_/r**2)[np.newaxis,:,:]
                

        print('B0m done')

        #B00 OLD
        B00 = -self.wig_red(0,0,0)*(2*U_ - 2*om(l_,0)**2 * V_ - r*dU_)*(-2*U + 2*om(l,0)**2 *V + \
                r*dU)
        B00 -= 2*r*(self.wig_red(-1,0,1) + self.wig_red(1,0,-1))*om(l_,0)*om(l,0) \
            *(U_ - V_ + r*dV_)*dV

        # B00 = np.tile(B00,(len_m,len_m_,1,1))

        B00 = (0.5*((-1)**np.abs(m_))*prefac)[:,:,:,np.newaxis] \
                * (B00/r**2)[np.newaxis,:,:]
        #B00 EXTRA
        B00_ = -(self.wig_red(-1,0,1) + self.wig_red(1,0,-1)) * om(l_,0)*om(l,0) * V*(-4*U_+2*(1+om(l_,0)**2)*V_+r*(dU_-2*dV_))
        B00_ += self.wig_red(0,0,0)*U*(2*U_-2*r*dU_-2*om(l_,0)**2 *(V_-r*dV_)+r*r*d2U_)

        #B00_ = np.tile(B00_,(len_m,len_m_,1,1))

        B00_ = (0.5*((-1)**np.abs(m_))*prefac)[:,:,:,np.newaxis] \
                * (B00_/r**2)[np.newaxis,:,:]

        print('B00 done')

        #B+- OLD
        Bpm = -r**2 * self.wig_red(0,0,0)*dU_*dU 
        Bpm += om(l_,0)*om(l,0)*(-2*(self.wig_red(-2,0,2)+self.wig_red(2,0,-2))*om(l_,2)*om(l,2)*V_*V \
                + self.wig_red(-1,0,1)*(U-V)*(U_ - V_ + r*dV_) + self.wig_red(1,0,-1) \
                *(U-V)*(U_ - V_ + r*dV_))

        # Bpm = np.tile(Bpm,(len_m,len_m_,1,1))

        Bpm = (0.5*((-1)**np.abs(m_))*prefac)[:,:,:,np.newaxis] \
                * (Bpm/r**2)[np.newaxis,:,:]
        #B0+- EXTRA
        Bpm_ = (self.wig_red(-1,0,1) + self.wig_red(1,0,-1)) * om(l_,0)*om(l,0) * V * (U_-V_+r*(-dU_+dV_))
        Bpm_ += self.wig_red(0,0,0) * r*r*U*d2U_

        # Bpm_ = np.tile(Bpm_,(len_m,len_m_,1,1))

        Bpm_ = (0.5*((-1)**np.abs(m_))*prefac)[:,:,:,np.newaxis] \
                * (Bpm_/r**2)[np.newaxis,:,:]

        print('Bpm done')

        Bmm += Bmm_
        B0m += B0m_
        B00 += B00_
        Bpm += Bpm_

        Bmm = Bmm.astype('float64')
        B0m = B0m.astype('float64')
        B00 = B00.astype('float64')
        Bpm = Bpm.astype('float64')
        
        #constructing the other two components of the kernel
        Bpp = parity_fac[:,:,:,np.newaxis]*Bmm
        B0p = parity_fac[:,:,:,np.newaxis]*B0m

        return Bmm,B0m,B00,Bpm,Bpp,B0p
