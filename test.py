import functions as fn
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ

e = 1e-7
x = np.linspace(-1+e,1-e,1e2)
l = 8
y1 = fn.P(x,l,1,3)
y2 = fn.P(x,l,1,3)
print (2.*l+1)/2 * integ.trapz(y1*y2, x)

#plt.plot(x,y1)
#plt.show()
