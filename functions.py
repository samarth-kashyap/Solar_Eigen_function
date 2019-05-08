import numpy as np

def deriv(y,x):
	"""returns derivative of y wrt x. same len as x and y"""
	if(len(x) != len(y)):	
		print("lengths dont match")
		exit()
	l = len(y)	
	ret = np.zeros(l)
	ret[0] = (y[1]-y[0]) / (x[1]-x[0])
	ret[-1] = (y[-1]-y[-2]) / (x[-1]-x[-2])
	for i in range(1,l-1):
		ret[i] = (y[i+1]-y[i-1]) / (x[i+1]-x[i-1])
	return ret
	
def nearest_index(array, value):
	"""finds index of object nearest to value in array"""
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx
