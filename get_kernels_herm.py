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

    def ret_kerns(self,smoothen=False):
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
        if(smoothen == True):
            npts = 300
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

        
        ###############################################################
        #no smoothing
        else: 
            r = self.r
            rho = self.rho
            Ui = self.Ui 
            Vi = self.Vi 
            Ui_ = self.Ui_ 
            Vi_ = self.Vi_ 

            dUi, dVi = np.gradient(Ui,r), np.gradient(Vi,r)
            d2Ui,d2Vi = np.gradient(dUi,r), np.gradient(dVi,r)
            dUi_, dVi_ = np.gradient(Ui_,r), np.gradient(Vi_,r)
            d2Ui_,d2Vi_ = np.gradient(dUi_,r), np.gradient(dVi_,r)
        # tstamp('load eigfiles')

        #################################################################3

        #making U,U_,V,V_,dU,dU_,dV,dV_,d2U,d2U_,d2V,d2V_ of same shape

        U = np.tile(Ui,(len_s,1))
        V = np.tile(Vi,(len_s,1))
        dU = np.tile(dUi,(len_s,1))
        dV = np.tile(dVi,(len_s,1))
        d2U = np.tile(d2Ui,(len_s,1))
        d2V = np.tile(d2Vi,(len_s,1))
        U_ = np.tile(Ui_,(len_s,1))
        V_ = np.tile(Vi_,(len_s,1))
        dU_ = np.tile(dUi_,(len_s,1))
        dV_ = np.tile(dVi_,(len_s,1))
        d2U_ = np.tile(d2Ui_,(len_s,1))
        d2V_ = np.tile(d2Vi_,(len_s,1))
        r = np.tile(r,(len_s,1))

        tstamp()

        # Initializing the omegas which will be used very often. Might cut down on some computational time

        om0 = om(l,0)
        om0_ = om(l_,0)
        om2 = om(l,2)
        om2_ = om(l_,2)
        om3 = om(l,3)
        om3_ = om(l_,3)

        #B-- EXPRESSION
        Bmm = self.wig_red(2,-2,0)*om0_*om2_*(V_*(3*U - 2*om0**2*V + 3*r*dU) - r*U*dV_)
        Bmm += self.wig_red(0,-2,2)*om0*om2*(V*(3*U_ - 2*om0_**2*V_ + 3*r*dU_) - r*U_*dV)
        Bmm += self.wig_red(1,-2,1)*om0_*om0*(3*U*V_ - 2*om0_**2*V_*V + om2_**2*V_*V + r*V*dU_ - r*U*dV_ - U_*U)
        Bmm += self.wig_red(1,-2,1)*om0*om0_*(3*U_*V - 2*om0**2*V*V_ + om2**2*V*V_ + r*V_*dU - r*U_*dV - U*U_)
        Bmm += self.wig_red(3,-2,-1)*om2_*om3_*om0*om0_*V_*V
        Bmm += self.wig_red(-1,-2,3)*om2*om3*om0_*om0*V*V_
        
        # np.save('prefac.npy',prefac)

        Bmm = 0.5*(((-1)**np.abs(1+m_))*prefac)[:,:,:,np.newaxis] \
                 * (Bmm/r**2)[np.newaxis,:,:]
    

        #tstamp('Bmm done')

        #B0- EXPRESSION
        B0m = self.wig_red(1,-1,0)*om0_*(4*om0**2*V_*V + 8*U*U_ - 5*om0**2*U_*V - 3*r*om0**2*V*dV_ + 2*r**2*dU*dV_ - r*om0**2*V_*dV + \
                r**2*V_*d2U + (-6.-2.*om0_**2 + om0**2)*U*V_ + r*U*(4*dV_ - r*d2V_))
        B0m += self.wig_red(0,-1,1)*om0*(4*om0_**2*V*V_ + 8*U_*U - 5*om0_**2*U*V_ - 3*r*om0_**2*V_*dV + 2*r**2*dU_*dV - r*om0**2*V*dV_ + \
                r**2*V*d2U_ + (-6. -2.*om0**2 + om0_**2)*U_*V + r*U_*(4*dV - r*d2V))
        B0m += self.wig_red(-1,-1,2)*om0*om0_*om2*(U*V_ + V*U_ - 4*V*V_ + 3*r*dV_*V + r*V_*dV)
        B0m += self.wig_red(2,-1,-1)*om0_*om0*om2_*(U_*V + V_*U - 4*V_*V + 3*r*dV*V_ + r*V*dV_)

        B0m = (0.25*((-1)**np.abs(m_))*prefac)[:,:,:,np.newaxis] \
                * (B0m/r**2)[np.newaxis,:,:]

        #tstamp('B0m done')
        
#        print(np.shape(self.wig_red(-1,-0,1)))
#        exit()

        #B00 EXPRESSION
        B00 = (self.wig_red(-1,0,1) + self.wig_red(1,0,-1))*om0_*om0*(2*(-1. + om0_**2)*V_*V - V_*(3*r*dU - 4*r*dV) - 2*r*U*dV_ - 2*r**2*dV_*dV)
        B00 += (self.wig_red(-1,0,1) + self.wig_red(1,0,-1))*om0*om0_*(2*(-1. + om0**2)*V*V_ - V*(3*r*dU_ - 4*r*dV_) - 2*r*U_*dV - 2*r**2*dV*dV_)
        B00 += self.wig_red(0,0,0)*(-26*om0_**2*U*V_ + 12*om0_**2*om0**2*V_*V + 2*om0_**2*r*U*dV_ - 2*r*dU_*U + r**2*dU_*dU + r**2*U*d2U_ + 14*U*U_)
        B00 += self.wig_red(0,0,0)*(-26*om0**2*U_*V + 12*om0**2*om0_**2*V*V_ + 2*om0**2*r*U_*dV - 2*r*dU*U_ + r**2*dU*dU_ + r**2*U_*d2U + 14*U_*U)

        B00 = (0.25*((-1)**np.abs(m_))*prefac)[:,:,:,np.newaxis] \
                * (B00/r**2)[np.newaxis,:,:]
        #tstamp('B00 done')

        #B+- EXPRESSION
        Bpm = self.wig_red(0,0,0)*(4*om0_**2*U*V_ - 4*om0_**2*om0**2*V_*V - 10*r*U*dU_ + 6*r*om0**2*V*dU_ - 3*r**2*dU_*dU + 4*r*om0_**2*U*dV_ - r**2*U*d2U_)
        Bpm += self.wig_red(0,0,0)*(4*om0**2*U_*V - 4*om0**2*om0_**2*V*V_ - 10*r*U_*dU + 6*r*om0_**2*V_*dU - 3*r**2*dU*dU_ + 4*r*om0**2*U_*dV - r**2*U_*d2U)
        Bpm += (self.wig_red(-2,0,2) + self.wig_red(2,0,-2))*(-2*om0_*om0*om2_*om2*V_*V)
        Bpm += (self.wig_red(-2,0,2) + self.wig_red(2,0,-2))*(-2*om0*om0_*om2*om2_*V*V_)
        Bpm += (self.wig_red(-1,0,1) + self.wig_red(1,0,-1))*om0_*om0*(V_*U - 2*om0_**2*V_*V + r*U*dV_ + U_*U)
        Bpm += (self.wig_red(-1,0,1) + self.wig_red(1,0,-1))*om0*om0_*(V*U_ - 2*om0**2*V*V_ + r*U_*dV + U*U_)


        Bpm = (0.25*((-1)**np.abs(m_))*prefac)[:,:,:,np.newaxis] \
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
            d2Ui,d2Vi = np.gradient(dUi,r), np.gradient(dVi,r)
            dUi_, dVi_ = np.gradient(Ui_,r), np.gradient(Vi_,r)
            d2Ui_,d2Vi_ = np.gradient(dUi_,r), np.gradient(dVi_,r)

        #########################################################################

        tstamp('load eigfiles')

        #making U,U_,V,V_,dU,dU_,dV,dV_,d2U,d2U_,d2V,d2V_ of same shape

        U = np.tile(Ui,(len_s,1))
        V = np.tile(Vi,(len_s,1))
        dU = np.tile(dUi,(len_s,1))
        dV = np.tile(dVi,(len_s,1))
        d2U = np.tile(d2Ui,(len_s,1))
        d2V = np.tile(d2Vi,(len_s,1))
        U_ = np.tile(Ui_,(len_s,1))
        V_ = np.tile(Vi_,(len_s,1))
        dU_ = np.tile(dUi_,(len_s,1))
        dV_ = np.tile(dVi_,(len_s,1))
        d2U_ = np.tile(d2Ui_,(len_s,1))
        d2V_ = np.tile(d2Vi_,(len_s,1))
        r = np.tile(r,(len_s,1))

        tstamp()

        # Initializing the omegas which will be used very often. Might cut down on some computational time

        om0 = om(l,0)
        om0_ = om(l_,0)
        om2 = om(l,2)
        om2_ = om(l_,2)
        om3 = om(l,3)
        om3_ = om(l_,3)

        #B-- EXPRESSION
        Bmm = self.wig_red(2,-2,0)*om0_*om2_*(V_*(3*U - 2*om0**2*V + 3*r*dU) - r*U*dV_)
        Bmm += self.wig_red(0,-2,2)*om0*om2*(V*(3*U_ - 2*om0_**2*V_ + 3*r*dU_) - r*U_*dV)
        Bmm += self.wig_red(1,-2,1)*om0_*om0*(3*U*V_ - 2*om0_**2*V_*V + om2_**2*V_*V + r*V*dU_ - r*U*dV_ - U_*U)
        Bmm += self.wig_red(1,-2,1)*om0*om0_*(3*U_*V - 2*om0**2*V*V_ + om2**2*V*V_ + r*V_*dU - r*U_*dV - U*U_)
        Bmm += self.wig_red(3,-2,-1)*om2_*om3_*om0*om0_*V_*V
        Bmm += self.wig_red(-1,-2,3)*om2*om3*om0_*om0*V*V_

        Bmm = 0.5*(((-1)**np.abs(1+m_))*prefac)[:,:,np.newaxis] \
                 * (Bmm/r**2)[np.newaxis,:,:]
        #tstamp('Bmm done')

        #B0- EXPRESSION
        B0m = self.wig_red(1,-1,0)*om0_*(4*om0**2*V_*V + 8*U*U_ - 5*om0**2*U_*V - 3*r*om0**2*V*dV_ + 2*r**2*dU*dV_ - r*om0**2*V_*dV + \
                r**2*V_*d2U + (-6.-2.*om0_**2 + om0**2)*U*V_ + r*U*(4*dV_ - r*d2V_))
        B0m += self.wig_red(0,-1,1)*om0*(4*om0_**2*V*V_ + 8*U_*U - 5*om0_**2*U*V_ - 3*r*om0_**2*V_*dV + 2*r**2*dU_*dV - r*om0**2*V*dV_ + \
                r**2*V*d2U_ + (-6. -2.*om0**2 + om0_**2)*U_*V + r*U_*(4*dV - r*d2V))
        B0m += self.wig_red(-1,-1,2)*om0*om0_*om2*(U*V_ + V*U_ - 4*V*V_ + 3*r*dV_*V + r*V_*dV)
        B0m += self.wig_red(2,-1,-1)*om0_*om0*om2_*(U_*V + V_*U - 4*V_*V + 3*r*dV*V_ + r*V*dV_)

        B0m = (0.25*((-1)**np.abs(m_))*prefac)[:,:,np.newaxis] \
                * (B0m/r**2)[np.newaxis,:]
        #tstamp('B0m done')
        
#        print(np.shape(self.wig_red(-1,-0,1)))
#        exit()

        #B00 EXPRESSION
        B00 = (self.wig_red(-1,0,1) + self.wig_red(1,0,-1))*om0_*om0*(2*(-1. + om0_**2)*V_*V - V_*(3*r*dU - 4*r*dV) - 2*r*U*dV_ - 2*r**2*dV_*dV)
        B00 += (self.wig_red(-1,0,1) + self.wig_red(1,0,-1))*om0*om0_*(2*(-1. + om0**2)*V*V_ - V*(3*r*dU_ - 4*r*dV_) - 2*r*U_*dV - 2*r**2*dV*dV_)
        B00 += self.wig_red(0,0,0)*(-26*om0_**2*U*V_ + 12*om0_**2*om0**2*V_*V + 2*om0_**2*r*U*dV_ - 2*r*dU_*U + r**2*dU_*dU + r**2*U*d2U_ + 14*U*U_)
        B00 += self.wig_red(0,0,0)*(-26*om0**2*U_*V + 12*om0**2*om0_**2*V*V_ + 2*om0**2*r*U_*dV - 2*r*dU*U_ + r**2*dU*dU_ + r**2*U_*d2U + 14*U_*U)

        B00 = (0.25*((-1)**np.abs(m_))*prefac)[:,:,np.newaxis] \
                * (B00/r**2)[np.newaxis,:]
        #tstamp('B00 done')

        #B+- EXPRESSION
        Bpm = self.wig_red(0,0,0)*(4*om0_**2*U*V_ - 4*om0_**2*om0**2*V_*V - 10*r*U*dU_ + 6*r*om0**2*V*dU_ - 3*r**2*dU_*dU + 4*r*om0_**2*U*dV_ - r**2*U*d2U_)
        Bpm += self.wig_red(0,0,0)*(4*om0**2*U_*V - 4*om0**2*om0_**2*V*V_ - 10*r*U_*dU + 6*r*om0_**2*V_*dU - 3*r**2*dU*dU_ + 4*r*om0**2*U_*dV - r**2*U_*d2U)
        Bpm += (self.wig_red(-2,0,2) + self.wig_red(2,0,-2))*(-2*om0_*om0*om2_*om2*V_*V)
        Bpm += (self.wig_red(-2,0,2) + self.wig_red(2,0,-2))*(-2*om0*om0_*om2*om2_*V*V_)
        Bpm += (self.wig_red(-1,0,1) + self.wig_red(1,0,-1))*om0_*om0*(V_*U - 2*om0_**2*V_*V + r*U*dV_ + U_*U)
        Bpm += (self.wig_red(-1,0,1) + self.wig_red(1,0,-1))*om0*om0_*(V*U_ - 2*om0**2*V*V_ + r*U_*dV + U*U_)

        Bpm = (0.25*((-1)**np.abs(m_))*prefac)[:,:,np.newaxis] \
                * (Bpm/r**2)[np.newaxis,:]
        #tstamp('Bpm done')

        #constructing the other two components of the kernel
        Bpp = parity_fac[:,:,np.newaxis]*Bmm
        Bp0 = parity_fac[:,:,np.newaxis]*B0m


        if(a_coeffkerns == True): 
            return rho,Bmm,B0m,B00,Bpm,Bp0,Bpp
        else:
            return Bmm,B0m,B00,Bpm,Bp0,Bpp
        
    def ret_splittingfunction_terms(self,smoothen = False):
        n,l,n_,l_ = self.n, self.l, self.n_, self.l_
        r_start , r_end = self.r_range
        nl = fn.find_nl(n,l)
        nl_ = fn.find_nl(n_,l_)

        len_s,__ = np.shape(self.ss_i)

        #Savitsky golay filter for smoothening
        window = 45  #must be odd
        order = 3

        if(nl == None or nl_ == None):
            print("Mode not found. Exiting."); exit()


        tstamp()
        om = np.vectorize(fn.omega,otypes=[float])
        wig_val = np.vectorize(fn.wig,otypes=[float])
        parity_fac = (-1)**(l+l_+self.s) #parity of selected modes
        prefac = 1./(4.* np.pi)/wig_val(l_,self.s,l,0,0,0)
        prefac[np.isinf(prefac)] = 0.0  #changing infinities to zeros. Splitting functions will be zeros Happens for l_+s+l=odd

        # tstamp('prefac computation')

        #EIGENFUCNTION DERIVATIVES

        ##########################################################3
        #smoothingR2 = 0.78
        #interpolation params
        if(smoothen == True):
            npts = 300
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

        
        ###############################################################
        #no smoothing
        else: 
            r = self.r
            rho = self.rho
            Ui = self.Ui 
            Vi = self.Vi 
            Ui_ = self.Ui_ 
            Vi_ = self.Vi_ 

            dUi, dVi = np.gradient(Ui,r), np.gradient(Vi,r)
            d2Ui,d2Vi = np.gradient(dUi,r), np.gradient(dVi,r)
            dUi_, dVi_ = np.gradient(Ui_,r), np.gradient(Vi_,r)
            d2Ui_,d2Vi_ = np.gradient(dUi_,r), np.gradient(dVi_,r)
        # tstamp('load eigfiles')

        #################################################################3

        #making U,U_,V,V_,dU,dU_,dV,dV_,d2U,d2U_,d2V,d2V_ of same shape

        U = np.tile(Ui,(len_s,1))
        V = np.tile(Vi,(len_s,1))
        dU = np.tile(dUi,(len_s,1))
        dV = np.tile(dVi,(len_s,1))
        d2U = np.tile(d2Ui,(len_s,1))
        d2V = np.tile(d2Vi,(len_s,1))
        U_ = np.tile(Ui_,(len_s,1))
        V_ = np.tile(Vi_,(len_s,1))
        dU_ = np.tile(dUi_,(len_s,1))
        dV_ = np.tile(dVi_,(len_s,1))
        d2U_ = np.tile(d2Ui_,(len_s,1))
        d2V_ = np.tile(d2Vi_,(len_s,1))
        r = np.tile(r,(len_s,1))

        tstamp()

        # Initializing the omegas which will be used very often. Might cut down on some computational time

        om0 = om(l,0)
        om0_ = om(l_,0)
        om2 = om(l,2)
        om2_ = om(l_,2)
        om3 = om(l,3)
        om3_ = om(l_,3)

        #B-- EXPRESSION
        Bmm = self.wig_red(2,-2,0)*om0_*om2_*(V_*(3*U - 2*om0**2*V + 3*r*dU) - r*U*dV_)
        Bmm += self.wig_red(0,-2,2)*om0*om2*(V*(3*U_ - 2*om0_**2*V_ + 3*r*dU_) - r*U_*dV)
        Bmm += self.wig_red(1,-2,1)*om0_*om0*(3*U*V_ - 2*om0_**2*V_*V + om2_**2*V_*V + r*V*dU_ - r*U*dV_ - U_*U)
        Bmm += self.wig_red(1,-2,1)*om0*om0_*(3*U_*V - 2*om0**2*V*V_ + om2**2*V*V_ + r*V_*dU - r*U_*dV - U*U_)
        Bmm += self.wig_red(3,-2,-1)*om2_*om3_*om0*om0_*V_*V
        Bmm += self.wig_red(-1,-2,3)*om2*om3*om0_*om0*V*V_
        
        # np.save('prefac.npy',prefac)

        Bmm = 0.5*(((-1)**np.abs(1))*prefac)[:,np.newaxis] * (Bmm/r**2)
    

        #tstamp('Bmm done')

        #B0- EXPRESSION
        B0m = self.wig_red(1,-1,0)*om0_*(4*om0**2*V_*V + 8*U*U_ - 5*om0**2*U_*V - 3*r*om0**2*V*dV_ + 2*r**2*dU*dV_ - r*om0**2*V_*dV + \
                r**2*V_*d2U + (-6.-2.*om0_**2 + om0**2)*U*V_ + r*U*(4*dV_ - r*d2V_))
        B0m += self.wig_red(0,-1,1)*om0*(4*om0_**2*V*V_ + 8*U_*U - 5*om0_**2*U*V_ - 3*r*om0_**2*V_*dV + 2*r**2*dU_*dV - r*om0**2*V*dV_ + \
                r**2*V*d2U_ + (-6. -2.*om0**2 + om0_**2)*U_*V + r*U_*(4*dV - r*d2V))
        B0m += self.wig_red(-1,-1,2)*om0*om0_*om2*(U*V_ + V*U_ - 4*V*V_ + 3*r*dV_*V + r*V_*dV)
        B0m += self.wig_red(2,-1,-1)*om0_*om0*om2_*(U_*V + V_*U - 4*V_*V + 3*r*dV*V_ + r*V*dV_)

        B0m = (0.25*((-1)**np.abs(0))*prefac)[:,np.newaxis] * (B0m/r**2)

        #tstamp('B0m done')
        
#        print(np.shape(self.wig_red(-1,-0,1)))
#        exit()

        #B00 EXPRESSION
        B00 = (self.wig_red(-1,0,1) + self.wig_red(1,0,-1))*om0_*om0*(2*(-1. + om0_**2)*V_*V - V_*(3*r*dU - 4*r*dV) - 2*r*U*dV_ - 2*r**2*dV_*dV)
        B00 += (self.wig_red(-1,0,1) + self.wig_red(1,0,-1))*om0*om0_*(2*(-1. + om0**2)*V*V_ - V*(3*r*dU_ - 4*r*dV_) - 2*r*U_*dV - 2*r**2*dV*dV_)
        B00 += self.wig_red(0,0,0)*(-26*om0_**2*U*V_ + 12*om0_**2*om0**2*V_*V + 2*om0_**2*r*U*dV_ - 2*r*dU_*U + r**2*dU_*dU + r**2*U*d2U_ + 14*U*U_)
        B00 += self.wig_red(0,0,0)*(-26*om0**2*U_*V + 12*om0**2*om0_**2*V*V_ + 2*om0**2*r*U_*dV - 2*r*dU*U_ + r**2*dU*dU_ + r**2*U_*d2U + 14*U_*U)

        B00 = (0.25*((-1)**np.abs(0))*prefac)[:,np.newaxis] * (B00/r**2)
        #tstamp('B00 done')

        #B+- EXPRESSION
        Bpm = self.wig_red(0,0,0)*(4*om0_**2*U*V_ - 4*om0_**2*om0**2*V_*V - 10*r*U*dU_ + 6*r*om0**2*V*dU_ - 3*r**2*dU_*dU + 4*r*om0_**2*U*dV_ - r**2*U*d2U_)
        Bpm += self.wig_red(0,0,0)*(4*om0**2*U_*V - 4*om0**2*om0_**2*V*V_ - 10*r*U_*dU + 6*r*om0_**2*V_*dU - 3*r**2*dU*dU_ + 4*r*om0**2*U_*dV - r**2*U_*d2U)
        Bpm += (self.wig_red(-2,0,2) + self.wig_red(2,0,-2))*(-2*om0_*om0*om2_*om2*V_*V)
        Bpm += (self.wig_red(-2,0,2) + self.wig_red(2,0,-2))*(-2*om0*om0_*om2*om2_*V*V_)
        Bpm += (self.wig_red(-1,0,1) + self.wig_red(1,0,-1))*om0_*om0*(V_*U - 2*om0_**2*V_*V + r*U*dV_ + U_*U)
        Bpm += (self.wig_red(-1,0,1) + self.wig_red(1,0,-1))*om0*om0_*(V*U_ - 2*om0**2*V*V_ + r*U_*dV + U*U_)


        Bpm = (0.25*((-1)**np.abs(0))*prefac)[:,np.newaxis] * (Bpm/r**2)

        #tstamp('Bpm done')

        # print(prefac.dtype,parity_fac.dtype,Bmm.dtype,B0m.dtype,B00.dtype,Bpm.dtype)

        #constructing the other two components of the kernel
        Bpp = parity_fac[:,np.newaxis]*Bmm
        Bp0 = parity_fac[:,np.newaxis]*B0m

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
