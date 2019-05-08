import numpy as np
import matplotlib.pyplot as plt
import os 
import functions as fn

r = np.loadtxt('r.dat')
eig_dir = os.getcwd()+'/eig_files'

for nl in range(100):
	n,l = fn.find_mode(nl)
	U = fn.load_U(n,l,eig_dir)
	V = fn.load_V(n,l,eig_dir)
	plt.figure()
	plt.plot(r,U,'r-')
	plt.plot(r,V,'b-')
	name = str(n)+'_'+str(l)
	print name
	plt.savefig(os.getcwd()+'/plots/'+name+'.png')
	plt.close()
exit()
