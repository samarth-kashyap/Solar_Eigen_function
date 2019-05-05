# Solar Eigen function reader (now used for Lorentz stress plotting kernels)

This repository contains the file related to reading the eigen functions.

The local directory must contain the files: 

* egvt.sfopal5h5
* sfopal5h5


Sequence:
1. Run $gfortran read_eigen.f90; ./a.out (for creating U,V, r, and rho binary data files)
2. Now, python scripts can be run to probe the data

The output are the following three files:

* n_and_l.dat: Contains the n,l values, i.e, the number of n's corresponding to a particular value of l. Note: n and l stars from 0 to n_max and l_max respectively.
* eigU.dat: Contains radial eigenvalues of shape (n_max+1)*(l_max+1) X Nr. Here Nr is the number of grids in the radial direction.
* eigV.dat: Contains angular eigenvalues of shape (n_max+1)*(l_max+1) X Nr. Here Nr is the number of grids in the radial direction.
* nl.dat: Contains the sequence of n,l for corresponding nl index

CAUTION: Running this code will generate data files of cumulative size ~ 1.1GB
