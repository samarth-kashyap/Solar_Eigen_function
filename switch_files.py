import os
import sys

if sys.argv[1] == 'down':
    os.rename('r.dat','r_full.dat')
    os.rename('rho.dat','rho_full.dat')
    os.rename('eig_files','eig_files_full')
    os.rename('w.dat','w_full.dat')

    os.rename('r_down.dat','r.dat')
    os.rename('rho_down.dat','rho.dat')
    os.rename('eig_files_down','eig_files')
    os.rename('w_down.dat','w.dat')

elif sys.argv[1] == 'up':
    os.rename('r.dat','r_down.dat')
    os.rename('rho.dat','rho_down.dat')
    os.rename('eig_files','eig_files_down')
    os.rename('w.dat','w_down.dat')

    os.rename('r_full.dat','r.dat')
    os.rename('rho_full.dat','rho.dat')
    os.rename('eig_files_full','eig_files')
    os.rename('w_full.dat','w.dat')

else:
    print 'invalid argument'
    sys.exit()
