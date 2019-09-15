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

    def __init__(self,n_,l_,m_,n,l,m,s,r,axis_symm = True):
        self.n = n
        self.l = l
        self.n_ = n_
        self.l_ = l_
        self.s = s
        r_full = np.loadtxt('r.dat')
        r_start, r_end = np.argmin(np.abs(r_full-r[0])),np.argmin(np.abs(r_full-r[-1]))+1
        self.r_range = r_start,r_end
        self.r = r
        #ss is m X m X s dim (outer)
        if(axis_symm == False):
            self.mm_, self.mm, self.ss_o = np.meshgrid(m_,m,s, indexing = 'ij')
        else:
            self.mm, self.ss_o = np.meshgrid(m,s, indexing = 'ij')
        #ss_in is s X r dim (inner)
        self.ss_i,__ = np.meshgrid(s,self.r, indexing = 'ij')
        
        #loading required functions
        eig_dir = (getcwd() + '/eig_files')
        Ui,Vi = fn.load_eig(n,l,eig_dir)
        Ui_,Vi_= fn.load_eig(n_,l_,eig_dir)
        rho = np.loadtxt('rho.dat')

        #slicing the radial function acoording to radial grids
        self.rho = rho[r_start:r_end]
        self.Ui = Ui[r_start:r_end]
        self.Vi = Vi[r_start:r_end]
        self.Ui_ = Ui_[r_start:r_end]
        self.Vi_ = Vi_[r_start:r_end]
        
    def wig_red_o(self,m1,m2,m3):
        '''3j symbol with upper row fixed (outer)'''
        wig_vect = np.vectorize(fn.wig,otypes=[float])
        return wig_vect(self.l_,self.ss_o,self.l,m1,m2,m3)

    def wig_red(self,m1,m2,m3):
        '''3j symbol with upper row fixed (inner)'''
        wig_vect = np.vectorize(fn.wig,otypes=[float])
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


        tstamp()
        om = np.vectorize(fn.omega,otypes=[float])
        parity_fac = (-1)**(l+l_+self.ss_o) #parity of selected modes
        prefac = 1./(4.* np.pi) * np.sqrt((2*l_+1.) * (2*self.ss_o+1.) * (2*l+1.) \
                    / (4.* np.pi)) * self.wig_red_o(-m_,m_-m,m)

        # tstamp('prefac computation')

        #EIGENFUCNTION DERIVATIVES

        ##########################################################3
        #smoothingR2 = 0.78
        #interpolation params
        # npts = 3000
        # r_new = np.linspace(np.amin(self.r),np.amax(self.r),npts)
        # self.ss_i,__ = np.meshgrid(self.s,r_new, indexing = 'ij')

        # Ui,dUi,d2Ui = fn.smooth(self.Ui,self.r,window,order,npts)
        # Vi,dVi,d2Vi = fn.smooth(self.Vi,self.r,window,order,npts)

        # Ui_,dUi_,d2Ui_ = fn.smooth(self.Ui_,self.r,window,order,npts)
        # Vi_,dVi_,d2Vi_ = fn.smooth(self.Vi_,self.r,window,order,npts)

        # rho_sm, __, __ = fn.smooth(self.rho,self.r,window,order,npts)
        # #re-assigning with smoothened variables
        # r = r_new
        # rho = rho_sm

        
        ###############################################################
        #no smoothing

        r = self.r
        rho = self.rho
        Ui = self.Ui 
        Vi = self.Vi 
        Ui_ = self.Ui_ 
        Vi_ = self.Vi_ 

        dUi, dVi = np.gradient(Ui,r), np.gradient(Vi,r)
        dUi_, dVi_ = np.gradient(Ui_,r), np.gradient(Vi_,r)
        d2Ui_,d2Vi_ = np.gradient(dUi_,r), np.gradient(dVi_,r)
        # tstamp('load eigfiles')

        #################################################################3

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

        # print(np.shape(V),np.shape(V_),np.shape(dU_),np.shape(r))

        #B-- EXPRESSION
        Bmm = self.wig_red(3,-2,-1)*om(l,0)*om(l_,0)*om(l_,2)*om(l_,3) * V*V_
        Bmm += self.wig_red(0,-2,2)*om(l,0)*om(l,2) * r*V*dU_
        Bmm += self.wig_red(1,-2,1)*om(l_,0)*om(l,0) * (-U*U_ + U*V_ + om(l_,2)**2 * V*V_ - r*U*dV_)
        Bmm += self.wig_red(2,-2,0)*om(l_,0)*om(l_,2) * (U*V_ + r*dU*V_ - r*U*dV_)  
        
        np.save('prefac.npy',prefac)

        Bmm = (((-1)**np.abs(1+m_))*prefac)[:,:,:,np.newaxis] \
                 * (Bmm/r**2)[np.newaxis,:,:]
    

        #tstamp('Bmm done')

        #B0- EXPRESSION
        B0m = self.wig_red(0,-1,1)*om(l,0) * (2*U*U_ + om(l_,2)**2*V*U_ + om(l_,0)**2*(-2*U*V_ - V*V_ + r*V*dV_) + r*(-U - V + r*dV)*dU_)
        B0m += self.wig_red(-1,-1,2)*om(l,0)*om(l_,0)*om(l,2) * V * (U_ - V_ + r*dV_)
        B0m += self.wig_red(2,-1,-1)*om(l,0)*om(l_,0)*om(l_,2) * (V*U_ - 3*V*V_ + r*V*dV_ + 2*r*dV*V_)
        B0m -= self.wig_red(1,-1,0)*om(l_,0) * (-2*U*U_ + om(l_,0)**2*V*U_ + om(l,0)**2*(-V*V_ + r*V*dV_) + U*(2*V_ + r*(dU_ - 2*dV_ + r*d2V_)))
        B0m = (0.5*((-1)**np.abs(m_))*prefac)[:,:,:,np.newaxis] \
                * (B0m/r**2)[np.newaxis,:,:]

        #tstamp('B0m done')
        
#        print(np.shape(self.wig_red(-1,-0,1)))
#        exit()

        #B00 EXPRESSION
        B00 = -(self.wig_red(-1,0,1)+self.wig_red(1,0,-1))*om(l_,0)*om(l,0) * (V*(-4*U_ + 2*(1+om(l_,0)**2)*V_ + r*(dU_ - 2*dV_)) + 2*r*dV*(U_ - V_ + r*dV_))
        B00 += self.wig_red(0,0,0) * ((6*U - 4*om(l,0)**2*V -2*r*dU)*U_ + 2*om(l_,0)**2*((-3*U+2*om(l,0)**2*V + r*dU)*V_ + r*U*dV_) + r*((-4*U + 2*om(l,0)**2*V + r*dU)*dU_ + r*U*d2U_))
        B00 = (0.5*((-1)**np.abs(m_))*prefac)[:,:,:,np.newaxis] \
                * (B00/r**2)[np.newaxis,:,:]
        #tstamp('B00 done')

        #B+- EXPRESSION
        Bpm = -2*(self.wig_red(-2,0,2)+self.wig_red(2,0,-2))*om(l_,0)*om(l,0)*om(l_,2)*om(l,2)*V*V_
        Bpm += (self.wig_red(-1,0,1)+self.wig_red(1,0,-1))*om(l_,0)*om(l,0) * (-r*V*dU_ + U*(U_-V_+r*dV_))
        Bpm += self.wig_red(0,0,0)*r*r * (-dU*dU_ + U*d2U_)
        Bpm = (0.5*((-1)**np.abs(m_))*prefac)[:,:,:,np.newaxis] \
                * (Bpm/r**2)[np.newaxis,:,:]

        #tstamp('Bpm done')

        # print(prefac.dtype,parity_fac.dtype,Bmm.dtype,B0m.dtype,B00.dtype,Bpm.dtype)

        #constructing the other two components of the kernel
        Bpp = parity_fac[:,:,:,np.newaxis]*Bmm
        Bp0 = parity_fac[:,:,:,np.newaxis]*B0m

        return Bmm,B0m,B00,Bpm,Bp0,Bpp
        
    def ret_kerns_axis_symm(self,smoothen = False, a_coeffkerns = False):
        n,l,m,n_,l_ = self.n, self.l, self.mm, self.n_, self.l_
        m_ = m
        r_start , r_end = self.r_range
        nl = fn.find_nl(n,l)
        nl_ = fn.find_nl(n_,l_)

        len_m, len_s = np.shape(self.ss_o)

        #Savitsky golay filter for smoothening
        window = 45  #must be odd
        order = 3

        if(nl == None or nl_ == None):
            print("Mode not found. Exiting."); exit()

        tstamp()
        om = np.vectorize(fn.omega,otypes=[float])
        parity_fac = (-1)**(l+l_+self.ss_o) #parity of selected modes
        if(a_coeffkerns == True):
            prefac = ((-1.)**l)/(4.* np.pi) * np.sqrt((2*l_+1.) * (2*self.ss_o+1.) * (2*l+1.) \
                    / (4.* np.pi)) * self.wig_red_o(-l,0,l) / l 
        else:
            prefac = 1./(4.* np.pi) * np.sqrt((2*l_+1.) * (2*self.ss_o+1.) * (2*l+1.) \
                    / (4.* np.pi)) * self.wig_red_o(-m,0,m)

        tstamp('prefac computation')

        #EIGENFUCNTION DERIVATIVES

        ###################################################################
        #smoothing

        #interpolation params

        if(smoothen == True):

            npts = 300      #should be less than the len(r) in r.dat
            r_new = np.linspace(np.amin(self.r),np.amax(self.r),npts)
            self.ss_i,__ = np.meshgrid(self.s,r_new, indexing = 'ij')

            Ui,dUi,d2Ui = fn.smooth(self.Ui,self.r,window,order,npts)
            Vi,dVi,d2Vi = fn.smooth(self.Vi,self.r,window,order,npts)

            Ui_,dUi_,d2Ui_ = fn.smooth(self.Ui_,self.r,window,order,npts)
            Vi_,dVi_,d2Vi_ = fn.smooth(self.Vi_,self.r,window,order,npts)

            rho_sm, __, __ = fn.smooth(self.rho,self.r,window,order,npts)
            #re-assigning with smoothened variables
            r = r_new
            rho = rho_sm
        

        ######################################################################
        #no smoothing

        else:

            r = self.r
            rho = self.rho
            Ui = self.Ui 
            Vi = self.Vi 
            Ui_ = self.Ui_ 
            Vi_ = self.Vi_ 

            dUi, dVi = np.gradient(Ui,r), np.gradient(Vi,r)
            dUi_, dVi_ = np.gradient(Ui_,r), np.gradient(Vi_,r)
            d2Ui_,d2Vi_ = np.gradient(dUi_,r), np.gradient(dVi_,r)

        #########################################################################

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
        Bmm = self.wig_red(3,-2,-1)*om(l,0)*om(l_,0)*om(l_,2)*om(l_,3) * V*V_
        Bmm += self.wig_red(0,-2,2)*om(l,0)*om(l,2) * r*V*dU_
        Bmm += self.wig_red(1,-2,1)*om(l_,0)*om(l,0) * (-U*U_ + U*V_ + om(l_,2)**2 * V*V_ - r*U*dV_)
        Bmm += self.wig_red(2,-2,0)*om(l_,0)*om(l_,2) * (U*V_ + r*dU*V_ - r*U*dV_)        
        Bmm = (((-1)**np.abs(1+m_))*prefac)[:,:,np.newaxis] \
                 * (Bmm/r**2)[np.newaxis,:,:]
        #tstamp('Bmm done')

        #B0- EXPRESSION
        B0m = self.wig_red(0,-1,1)*om(l,0) * (2*U*U_ + om(l_,2)**2*V*U_ + om(l_,0)**2*(-2*U*V_ - V*V_ + r*V*dV_) + r*(-U - V + r*dV)*dU_)
        B0m += self.wig_red(-1,-1,2)*om(l,0)*om(l_,0)*om(l,2) * V * (U_ - V_ + r*dV_)
        B0m += self.wig_red(2,-1,-1)*om(l,0)*om(l_,0)*om(l_,2) * (V*U_ - 3*V*V_ + r*V*dV_ + 2*r*dV*V_)
        B0m -= self.wig_red(1,-1,0)*om(l_,0) * (-2*U*U_ + om(l_,0)**2*V*U_ + om(l,0)**2*(-V*V_ + r*V*dV_) + U*(2*V_ + r*(dU_ - 2*dV_ + r*d2V_)))
        B0m = (0.5*((-1)**np.abs(m_))*prefac)[:,:,np.newaxis] \
                * (B0m/r**2)[np.newaxis,:]
        #tstamp('B0m done')
        
#        print(np.shape(self.wig_red(-1,-0,1)))
#        exit()

        #B00 EXPRESSION
        B00 = -(self.wig_red(-1,0,1)+self.wig_red(1,0,-1))*om(l_,0)*om(l,0) * (V*(-4*U_ + 2*(1+om(l_,0)**2)*V_ + r*(dU_ - 2*dV_)) + 2*r*dV*(U_ - V_ + r*dV_))
        B00 += self.wig_red(0,0,0) * ((6*U - 4*om(l,0)**2*V -2*r*dU)*U_ + 2*om(l_,0)**2*((-3*U+2*om(l,0)**2*V + r*dU)*V_ + r*U*dV_) + r*((-4*U + 2*om(l,0)**2*V + r*dU)*dU_ + r*U*d2U_))
        B00 = (0.5*((-1)**np.abs(m_))*prefac)[:,:,np.newaxis] \
                * (B00/r**2)[np.newaxis,:]
        #tstamp('B00 done')

        #B+- EXPRESSION
        Bpm = -2*(self.wig_red(-2,0,2)+self.wig_red(2,0,-2))*om(l_,0)*om(l,0)*om(l_,2)*om(l,2)*V*V_
        Bpm += (self.wig_red(-1,0,1)+self.wig_red(1,0,-1))*om(l_,0)*om(l,0) * (-r*V*dU_ + U*(U_-V_+r*dV_))
        Bpm += self.wig_red(0,0,0)*r*r * (-dU*dU_ + U*d2U_)
        Bpm = (0.5*((-1)**np.abs(m_))*prefac)[:,:,np.newaxis] \
                * (Bpm/r**2)[np.newaxis,:]
        #tstamp('Bpm done')

        #constructing the other two components of the kernel
        Bpp = parity_fac[:,:,np.newaxis]*Bmm
        Bp0 = parity_fac[:,:,np.newaxis]*B0m


        if(a_coeffkerns == True): 
            return rho,Bmm,B0m,B00,Bpm,Bp0,Bpp
        else:
            return Bmm,B0m,B00,Bpm,Bp0,Bpp
        

        
    def Tkern(self,s):   
        n,l,n_,l_ = self.n, self.l, self.n_, self.l_
        r = self.r
        rho = self.rho
        Ui = self.Ui 
        Vi = self.Vi 
        Ui_ = self.Ui_ 
        Vi_ = self.Vi_ 

        len_s = len(s)

        tstamp()
        U = np.tile(Ui,(len_s,1))
        V = np.tile(Vi,(len_s,1))
        U_ = np.tile(Ui_,(len_s,1))
        V_ = np.tile(Vi_,(len_s,1))
        #tstamp('Tiling in Tkerns ends')

        wig_calc = np.vectorize(fn.wig,otypes=[float])

        ss,rr = np.meshgrid(s,r,indexing='ij')
        

        tstamp()
        T_s_r = (1-(-1)**(l_+l+ss))*fn.omega(l_,0)*fn.omega(l,0) \
            *wig_calc(l_,ss,l,-1,0,1)*(U_*V+V_*U-U_*U-0.5*V*V_*(l*(l+1) + \
            l_*(l_+1)-ss*(ss+1)))/rr            
        #tstamp('Computing T_s_r ends')

        return T_s_r
