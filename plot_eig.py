import numpy as np
import matplotlib.pyplot as plt
import os 
import functions as fn

r = np.loadtxt('r.dat')
#eig_dir = os.getcwd()+'/eig_files'
#sd = os.getcwd()+'/plots' #storage directory
#if (os.path.exists(sd) == False):	os.mkdir(sd)

#for nl in range(100):
#	n,l = fn.find_mode(nl)
#	U,V = fn.load_eig(n,l,eig_dir)
#	plt.figure()
#	plt.plot(r,U,'r-')
#	plt.plot(r,V,'b-')
#	name = str(n)+'_'+str(l)
#	print name
#	plt.savefig(sd+'/'+name+'.png')
#	plt.close()
#exit()
U,V = fn.load_eig(1,60,'eig_files')
plt.plot(r,U)
plt.show()
