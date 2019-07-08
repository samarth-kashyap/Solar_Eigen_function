import numpy as np
import functions as fn
import matplotlib.pyplot as plt
import scipy.special
import scipy.integrate

l = 75
m = np.arange(-l,l+1)

plt.plot(m,np.vectorize(fn.P_a)(m,l,0))
plt.show()


