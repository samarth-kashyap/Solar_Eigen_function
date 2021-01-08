import numpy as np
from scipy.interpolate import interp1d
# import functions as fn
#all quantities in cgs
#M_sol = 1.989e33 #gn,l = 0,200
#R_sol = 6.956e10 #cm
#B_0 = 10e5 #G
#OM = np.sqrt(4*np.pi*R_sol*B_0**2/M_sol)

OM = np.loadtxt('OM.dat') #importing normalising frequency value from file (in Hz (cgs))
r = np.loadtxt('/scratch/g.samarth/Solar_Eigen_function/r.dat')

w_file = np.loadtxt('w_samarth.dat')

r_w = (w_file)[0]
w_1 = -(w_file)[1]
w_3 = -(w_file)[2]
w_5 = -(w_file)[3]

#subtracting 440 nHz. 

# w_1 -= (r_w*440.0/fn.gam(1))

w1_interp = interp1d(r_w,w_1,kind='cubic',bounds_error=False,fill_value=0)
w3_interp = interp1d(r_w,w_3,kind='cubic',bounds_error=False,fill_value=0)
w5_interp = interp1d(r_w,w_5,kind='cubic',bounds_error=False,fill_value=0)

w = np.zeros((3,len(r)))

w1_f = w1_interp(r)
w3_f = w3_interp(r)
w5_f = w5_interp(r)

w[0] = w1_f
w[1] = w3_f
w[2] = w5_f

w *= (1e-9 / OM) #w normalising w to natural units of velocity R_sol*OM

np.savetxt('w.dat',w)
