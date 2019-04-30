# Solar Eigen function reader

This repository contains the file related to reading the eigen functions.

The local directory must contain the files: 

* egvt.sfopal5h5
* sfopal5h5

Run the code 'read_eigen.f90' by:

$ gfortran read_eigen.f90

$ ./a.out

The output are the following three files:

* n_and_l.dat: Contains the n,l values, i.e, the number of n's corresponding to a particular value of l. Note: n and l stars from 0 to n_max and l_max respectively.
* eigU.dat: Contains radial eigenvalues of shape (n_max+1)*(l_max+1) X Nr. Here Nr is the number of grids in the radial direction.
* eigV.dat: Contains angular eigenvalues of shape (n_max+1)*(l_max+1) X Nr. Here Nr is the number of grids in the radial direction.

CAUTION: Running this code will generate data files of cumulative size ~ 1.1GB
