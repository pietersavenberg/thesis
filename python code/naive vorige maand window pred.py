from scipy import interpolate
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import *
from scipy import optimize
from scipy import linalg
import matplotlib.pyplot as plt


  
def NMSE(yvalid,ypred):
    nmse = 0
    try:
        for i in range(yvalid.shape[1]):
            nmse += mean_squared_error(yvalid[:,i],ypred[:,i]) /(np.std(yvalid[:,i])**2)
    except:
        nmse += mean_squared_error(yvalid,ypred) /(np.std(yvalid)**2)

    return nmse
    
def hitrate(yvalid,ypred):
    hitrate = 0.
    try:
        for i in range(yvalid.shape[0]):
            for j in range(yvalid.shape[1]):
                if yvalid[i,j]*ypred[i,j] > 0:
                    hitrate += 1. 
                elif ypred[i,j] == 0 and yvalid[i,j] >0:
                    hitrate += 1.
        return hitrate/(yvalid.shape[0]*yvalid.shape[1])
    except:
        for i in range(yvalid.shape[0]):
            if yvalid[i]*ypred[i] > 0:
                hitrate += 1. 
            elif ypred[i] == 0 and yvalid[i] >0:
                hitrate += 1. 
        return (hitrate/(yvalid.shape[0]))
        
def score(yvalid,ypred):
    return np.sqrt(hitrate(yvalid,ypred)/NMSE(yvalid,ypred))
    
def rchange(arr):
    change = np.zeros(arr.size-1)
    for i in range(change.size):
        if arr[i] != 0 :
            change[i] =( arr[i+1] - arr[i])/arr[i]
        if arr[i] == 0 and arr[i+1] !=0:
            change[i] = 1
        else:
            change[i] = 0
    return change


y = np.load('y.npy')
yalsdet = np.load('yalsdet.npy')

yvalidlist = np.zeros(15)
ypredlist = np.zeros(15)

n_windows = 4
for i in range(n_windows):
    yalsdetwindow = yalsdet[522+13*i:535+13*i]
    ywindow = y[522+13*i:535+13*i]
        
    yvalidlist = np.vstack((yvalidlist,ywindow))
    ypredlist = np.vstack((ypredlist,yalsdetwindow))

yvalidlist = yvalidlist[1:] #remove zeros
ypredlist = ypredlist[1:]

ax = range(522,yvalidlist.shape[0]+522,1)


for i in range(yvalidlist.shape[1]):
    plt.figure(i+1)
    #plt.fill_between(ax, ypredlist[:,i]-1.96*(sigmalist[:,i]), ypredlist[:,i]+1.96*(sigmalist[:,i]),alpha = .5,  facecolor='blue', interpolate=True,label='95% confidence interval')
    plt.plot(ax, ypredlist[:,i],'b-',label=u'Predictions')
    plt.plot(ax, yvalidlist[:,i],'r-',label=u'Observations')
    plt.xlabel('time (weeks)')
    plt.ylabel('return')
    plt.vlines(range(12+522,len(ax)+522,13),-0.1,0.1,'y')
    plt.title(('Country',i))
    plt.legend(loc='upper right')
    fig = plt.gcf()
    fig.set_size_inches(15,8)

plt.show()

for i in range(15):
    print "Country ",i," NMSE :", NMSE(yvalidlist[:,i],ypredlist[:,i])
    print "Country ",i," hitrate :", hitrate(yvalidlist[:,i],ypredlist[:,i])

print "total NMSE :", NMSE(yvalidlist,ypredlist)
print "mean hitrate :", hitrate(yvalidlist,ypredlist)

#np.save('ytrain', ytrain)
np.save('ypredlist', ypredlist)
np.save('yvalidlist', yvalidlist)