import os
import sys

if(os.path.isfile('r.dat') == False):
    print('no r.dat found. exiting')
    sys.exit()

if sys.argv[1] == 'down':
    if(os.path.isfile('r_down.dat') == False):
        print('already in downsample mode')
        sys.exit()
    os.system('mv r.dat r_full.dat')
    os.system('mv rho.dat rho_full.dat')
    os.system('mv eig_files eig_files_full')

    os.system('mv r_down.dat r.dat')
    os.system('mv rho_down.dat rho.dat')
    os.system('mv eig_files_down eig_files')

elif sys.argv[1] == 'up':
    if(os.path.isfile('r_full.dat') == False):
        print('already in upsampled mode')
        sys.exit()
    os.system('mv r.dat r_down.dat')
    os.system('mv rho.dat rho_down.dat')
    os.system('mv eig_files eig_files_down')

    os.system('mv r_full.dat r.dat')
    os.system('mv rho_full.dat rho.dat')
    os.system('mv eig_files_full eig_files')

else:
    print('invalid argument')
    sys.exit()

os.system('python write_w.py')
