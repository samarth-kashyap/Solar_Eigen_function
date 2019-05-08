import numpy as np
import os
#import matplotlib.pyplot as plt

cwd = os.getcwd()
sd = cwd + '/eig_files' #storage directory

if(os.path.isdir(sd) == False):	os.mkdir(sd)

nl = np.loadtxt('nl.dat')
U_list = np.loadtxt('eigU.dat')
V_list = np.loadtxt('eigV.dat')
mode_count = len(nl)

l_pres = 0
l_prev = -1
for i in range(mode_count):
	l_pres = int(nl[i][1])
	if(l_prev != l_pres):
		print('l = '+str(l_pres))
		l_prev = l_pres
	Uname = 'U'+str(i)
	Vname = 'V'+str(i)
	U, V = U_list[i], V_list[i]
	np.savetxt(sd + '/' + Uname+'.dat', U)
	np.savetxt(sd + '/' + Vname+'.dat', V)
