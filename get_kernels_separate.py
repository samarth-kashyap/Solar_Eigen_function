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

    def __init__(self,n_,l_,m_,n,l,m,s,r_start,r_end):
        self.n = n
        self.l = l
        self.n_ = n_
        self.l_ = l_
        r_full = np.loadtxt('r.dat')
        self.r = r_full[r_start:r_end]
        #ss is m X m X s dim (outer)
        self.mm_, self.mm, self.ss_o = np.meshgrid(m_,m,s, indexing = 'ij')
        #ss_in is s X r dim (inner)
        self.ss_i,__ = np.meshgrid(s,self.r, indexing = 'ij')
        self.r_range = r_start,r_end

    def wig_red_o(self,m1,m2,m3):
        '''3j symbol with upper row fixed (outer)'''
        wig_vect = np.vectorize(fn.wig)
        return wig_vect(self.l_,self.ss_o,self.l,m1,m2,m3)

    def wig_red(self,m1,m2,m3):
        '''3j symbol with upper row fixed (inner)'''
        wig_vect = np.vectorize(fn.wig)
        return wig_vect(self.l_,self.ss_i,self.l,m1,m2,m3)

    def ret_kerns(self):
        n,l,m,n_,l_,m_ = self.n, self.l, self.mm, self.n_, self.l_, self.mm_
        r_start , r_end = self.r_range
        nl = fn.find_nl(n,l)
        nl_ = fn.find_nl(n_,l_)

        len_m, len_m_, len_s = np.shape(self.ss_o)

        #Savitsky golay filter for smoothening
        window = 45  #must be odd
        order = 3

        if(nl == None or nl_ == None):
            print("Mode not found. Exiting."); exit()

        #loading required functions
        eig_dir = (getcwd() + '/eig_files')
        Ui,Vi = fn.load_eig(n,l,eig_dir)
        Ui_,Vi_= fn.load_eig(n_,l_,eig_dir)
        rho = np.loadtxt('rho.dat')

        #slicing the radial function acoording to radial grids
        r = self.r
        rho = rho[r_start:r_end]
        Ui = Ui[r_start:r_end]
        Vi = Vi[r_start:r_end]
        Ui_ = Ui_[r_start:r_end]
        Vi_ = Vi_[r_start:r_end]

        tstamp()
        om = np.vectorize(fn.omega)
        parity_fac = (-1)**(l+l_+self.ss_o) #parity of selected modes
        prefac = 1./(4.* np.pi) * np.sqrt((2*l_+1.) * (2*self.ss_o+1.) * (2*l+1.) \
                    / (4.* np.pi)) * self.wig_red_o(-m_,m_-m,m)
        tstamp('prefac computation')

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
        tstamp('load eigfiles')

        #making U,U_,V,V_,dU,dU_,dV,dV_,d2U,d2U_,d2V,d2V_ of same shape

        U = np.tile(Ui,(len_s,1))
        V = np.tile(Vi,(len_s,1))
        dU = np.tile(dUi,(len_s,1))
        dV = np.tile(dVi,(len_s,1))
        U_ = np.tile(Ui_,(len_s,1))
        V_ = np.tile(Vi_,(len_s,1))
        dU_ = np.tile(dUi_,(len_s,1))
        dV_ = np.tile(dVi_,(len_s,1))
        d2U_ = np.tile(d2Ui_,(len_s,1))
        d2V_ = np.tile(d2Vi_,(len_s,1))
        r = np.tile(r,(len_s,1))

        tstamp()

        #B-- EXPRESSION
        Bmm = -r*(self.wig_red(0,-2,2)*om(l,0)*om(l,2)*V*dU_ + self.wig_red(2,-2,0)*om(l_,0)* \
                om(l_,2)*V_*dU)
        Bmm += self.wig_red(1,-2,1)*om(l_,0)*om(l,0)*(U-V)*(U_ - V_ + r*dV_)
        Bmm = (((-1)**np.abs(m_))*prefac)[:,:,:,np.newaxis] \
                 * (Bmm/r**2)[np.newaxis,:,:]
        #B-- EXTRA
        Bmm_ = om(l_,0)*(self.wig_red(2,-2,0)*om(l_,2)*U*(V_ - r*dV_) + om(l,0)*V \
                *(self.wig_red(3,-2,-1)*om(l_,2)*om(l_,3)*V_ + self.wig_red(1,-2,1) \
                *(-U_ + V_ + om(l_,2)**2 *V_ - r*dV_)))
        Bmm_ = (((-1)**np.abs(1+m_))*prefac)[:,:,:,np.newaxis] \
                 * (Bmm_/r**2)[np.newaxis,:,:]

        tstamp('Bmm done')

        #B0- EXPRESSION
        B0m = self.wig_red(1,-1,0)*om(l_,0)*(U - (om(l,0)**2)*V)*(U_ - V_ + r*dV_)
        B0m += om(l,0)*(om(l_,0)*(self.wig_red(-1,-1,2)*om(l,2)*V*(U_ - V_ + r*dV_) \
            + 2*r*self.wig_red(2,-1,-1)*om(l_,2)*V_*dV) + self.wig_red(0,-1,1) \
            *((U-V)*(2*U_ - 2*(om(l_,0)**2)*V_ - r*dU_) + r**2 * dU_*dV))

        B0m = (0.5*((-1)**np.abs(m_))*prefac)[:,:,:,np.newaxis] \
                * (B0m/r**2)[np.newaxis,:,:]
                
        #B0- EXTRA
        B0m_ = om(l,0)*V*(self.wig_red(2,-1,-1)*om(l_,0)*om(l_,2)*(U_ - 3*V_ + r*dV_) \
                + self.wig_red(0,-1,1)*((2+om(l_,0)**2)*U_ - 2*r*dU_ + om(l_,0)**2 \
                *(-3*V_ + r*dV_)))
        B0m_ += self.wig_red(1,-1,0)*om(l_,0)*U*(U_ - V_ - r*(dU_ - dV_ + r*d2V_))
        B0m_ = (0.5*((-1)**np.abs(m_))*prefac)[:,:,:,np.newaxis] \
                * (B0m_/r**2)[np.newaxis,:,:]
        tstamp('B0m done')

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

        tstamp('B00 done')

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

        tstamp('Bpm done')

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
        Bp0 = parity_fac[:,:,:,np.newaxis]*B0m

        return Bmm,B0m,B00,Bpm,Bp0,Bpp
