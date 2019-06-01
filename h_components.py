import numpy as np
import functions as fn

mu = np.array([-1,0,1])
nu = np.array([-1,0,1])
s = np.array([0,1,2])
t = np.arange(-np.max(np.abs(s)),np.max(np.abs(s))+1,1)

mumu,nunu,ss,tt = np.meshgrid(mu,nu,s,t,indexing='ij')

wig_calc = np.vectorize(fn.wig)

wig1 = wig_calc(1,ss,1,mumu,-(mumu+nunu),nunu)
wig2 = wig_calc(1,ss,1,0,tt,0)

H = np.sqrt((9/(4*np.pi))*(2*ss+1))*wig1*wig2

r = np.loadtxt('r.dat')

r = r[-700:]

HH = H[:,:,:,:,np.newaxis]*r[np.newaxis,:]

HH = HH.astype('float64')

        



