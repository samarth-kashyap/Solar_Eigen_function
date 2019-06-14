# Solar Lorentz Stress Kernel Plotter

Repository contains fortran and python script for evaluation and plotting of solar Lorentz stress kernel.

Local directory should contain the following data files:
* egvt.sfopal5h5: binary file containing eigenfunctions in compact form
* sfopal5h5: binary file containing r gridpoints and rho values

Sequence:
1. Run read_eig.f90. Reads egvt.sfopal5h5 and saves eigenfunctions in separate files. 
2. Run mince_eig.py. Reads eigenfunctions from above created files and saves individual modes in new directory eig_files.
3. Run plot_kern.py. Plots required component of kernel for required modes.

* Python Libraries required: numpy, sympy, matplotlob, os
* Python version used: 2.7.15
* Fortran version used: GNU Fortran 7.4.0
* CAUTION: Running this code will generate data files of cumulative size ~ 2.5GB
