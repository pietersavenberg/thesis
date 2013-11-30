import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import *
import matplotlib.pyplot as plt
from matplotlib.colors import *
from scipy import interpolate


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
        return hitrate/(yvalid.shape[0]*yvalid.shape[1])
    except:
        for i in range(yvalid.shape[0]):
            if yvalid[i]*ypred[i] > 0:
                hitrate += 1. 
        return (hitrate/(yvalid.shape[0]))
        
def score(yvalid,ypred):
    return np.sqrt(hitrate(yvalid,ypred)/NMSE(yvalid,ypred))
                
#det = 15 #aantal determinanten die we overhouden (computatie sneller), 15 van yalsdet + 42*2 van DSdet

y = np.load('y.npy')
yalsdet = np.load('yalsdet.npy')
DSdet = np.load('det.npy')
tijdquart = np.load('a.npy')#tijdsas voor kwartaal data: - 92 0 tem 7029 = 15 mei 2012
tijdmaand = np.load('b.npy')#tijdsas voor maandelijkse data: - 31 0 28 tem 7029
qdata = np.load('q.npy') #variable y-as per determinant voor kwartaal data
mdata = np.load('m.npy') #variable y-as per determinant voor maandelijkse data


tijdtrainvalid = range(-7,7035,7)#tijdsas waarvoor cubic spline waarden moet zoeken voor de determinanten, loopt tem 7028 = 14 mei 2012

#interpolatie naar wekelijks
x = np.zeros(len(tijdtrainvalid)-1)

for i in range(DSdet.shape[0]):
    changedet = rchange(DSdet[i])
    x = np.vstack((x, np.vstack((DSdet[i,1:],changedet))))
    
for i in range(qdata.shape[0]):
    tck = interpolate.splrep(tijdquart,qdata[i],s=0)
    ynew = interpolate.splev(tijdtrainvalid,tck,der=0)
    change = rchange(ynew)
    ynew= ynew[1:] #na de rchange hebben we de waarde voor -7 ni meer nodig
    x = np.vstack((x, np.vstack((ynew,change))))

for i in range(mdata.shape[0]):
    tck = interpolate.splrep(tijdmaand,mdata[i],s=0)
    ynew = interpolate.splev(tijdtrainvalid,tck,der=0)
    change = rchange(ynew)
    ynew= ynew[1:] #na de rchange hebben we de waarde voor -7 ni meer nodig
    x = np.vstack((x, np.vstack((ynew,change))))  

x = x[1:] #remove first row zeros
x = x.T
x = np.hstack((yalsdet,x))

#train/validatie scheiden
todelete = []
yvalid,Xvalid = [],[]
tijdvalid = []
for i in range(1,17):
    todelete += range(60*i,60*i+8)
    yvalid += list(y[60*i:60*i+8])
    Xvalid +=list(x[60*i:60*i+8])
    tijdvalid += tijdtrainvalid[60*i:60*i+8]
 
Xvalid = np.asarray(Xvalid)
yvalid = np.asarray(yvalid)
Xtrain = np.delete(x,todelete,axis = 0)
ytrain = np.delete(y,todelete,axis = 0)

#X is als volgt opgebouwd: eerst 15 yalsdet, dan 2*42 DSdet, dan 2*213 kwartaaldata, dan 2*490 maandelijkse data = totaal 1505
#Xvalid = Xvalid[:,:det]
#Xtrain = Xtrain[:,:det]

Ridge = Ridge(alpha=0.1, fit_intercept=True, normalize=True, copy_X=True, max_iter=None, tol=0.001, solver='auto')
Ridge.fit(Xtrain, ytrain) 
ypred  = Ridge.predict(Xvalid)

ax = range(1,129,1)

for i in range(15):
    plt.figure(i+1)
    plt.plot(ax, yvalid[:,i])
    plt.plot(ax, ypred[:,i])
    plt.xlabel('time (weeks)')
    plt.ylabel('return')
    plt.vlines(range(8,136,8),-0.1,0.1,'r')
    plt.title(('Country',i))
    fig = plt.gcf()
    fig.set_size_inches(15,8)

plt.show()

for i in range(15):
    print "Country ",i," NMSE :", NMSE(yvalid[:,i],ypred[:,i])
    print "Country ",i," hitrate :", hitrate(yvalid[:,i],ypred[:,i])

print "total NMSE :", NMSE(yvalid,ypred)
print "total hitrate :", hitrate(yvalid,ypred)