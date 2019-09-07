import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.patches import Rectangle

ytickinfo = np.loadtxt('ytickinfo.dat')

labels,location = ytickinfo[:,0],ytickinfo[:,1]

r_start = 0.68
r_end = 1.0
npts = 300

nl_list = np.loadtxt('../nl.dat')
lmax_arr = np.zeros(32)

for i in range(0,32):
    lmax_arr[i] = np.amax(nl_list[nl_list[:,0]==i]) + lmax_arr[i-1]

#sys.exit()

r = np.linspace(r_start,r_end,npts)

nrg = 100 #no of grid in r from end. Basically array[:,-nrg:]

xlabel, xloc = r[-nrg:], np.linspace(0,nrg-1,nrg)
xlabel = np.round(xlabel,3)

# plt.yticks(location[:-1:5], labels[:-1:5])
# plt.xticks(xloc[::15],xlabel[::15])

fig = plt.figure(figsize=(10,10)) 
ax = fig.add_subplot(111) 

plt.yticks(lmax_arr, labels)
plt.xticks(xloc[::15],xlabel[::15])

ax.grid(True,axis='x',linestyle='--',alpha=0.5,color='black')
ax.grid(True,axis='y',linewidth=1,color='black')

for axi in (ax.xaxis, ax.yaxis):
    for tic in axi.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False

for i in range(0,10):
    plt.text(-2.7,location[i]-50,int(labels[i]))

for i in range(10,30,5):
    plt.text(-3,location[i],int(labels[i]))

for i in range(0,len(xlabel),15):
    plt.text(xloc[i]-2,-250,xlabel[i])

plt.ylabel('Cumulative $n_{\ell}$',labelpad = 30,fontsize=14)
plt.xlabel('$r/R_{\odot}$',labelpad = 30,fontsize=14)
ax.add_patch(Rectangle((0, 0), 200, 600, facecolor="grey",alpha=0.3))
#plt.savefig('gridfig1.pdf')
plt.text(43,4700,'(a)',fontsize=14)
plt.show('Block')

fig = plt.figure(figsize=(10,10)) 
ax = fig.add_subplot(111) 

y = np.linspace(0,300,301)
yloc = np.append(y,y+300)
ylabel = np.append(y,y)
ylabel = ylabel.astype(int)

plt.yticks(yloc[::20], ylabel[::20])
plt.xticks(xloc[::15],xlabel[::15])
fs = 14
plt.grid(True,axis='y',linewidth=1.,color='black')
plt.grid(True,axis='x',linestyle='--',alpha=0.5,color='black')
ax.annotate('n=0', xy=(-0.08, 0.25), xytext=(-0.13, 0.231), xycoords='axes fraction', 
            fontsize=fs, ha='center', va='bottom',
            arrowprops=dict(arrowstyle='-[, widthB=6.2, lengthB=0.3', lw=2.0))


ax.annotate('n=1', xy=(-0.08, 0.75), xytext=(-0.13, 0.731), xycoords='axes fraction', 
            fontsize=fs, ha='center', va='bottom',
            arrowprops=dict(arrowstyle='-[, widthB=6.2, lengthB=0.3', lw=2.0))

ax.set_facecolor('#d8dcd6')

plt.xlabel('$r/R_{\odot}$',fontsize=14)
plt.text(43,620,'(b)',fontsize=14)
#plt.savefig('gridfig2.pdf')