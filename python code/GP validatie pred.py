import numpy as np
from sklearn.gaussian_process import GaussianProcess
from sklearn.metrics import *
import matplotlib.pyplot as plt
from matplotlib.colors import *
#import mpl_toolkits.mplot3d 
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
                
det = 5 #aantal determinanten die we overhouden (computatie sneller), 15 van yalsdet + 42*2 van DSdet + 2*11 qbelgium + 2*19 qdenmark + 2*13 qfinland + paar van qfrance

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
Xvalid = Xvalid[:,:det]
Xtrain = Xtrain[:,:det]

n_features = Xtrain.shape[1]

theta0 = np.array((1e-3,10,1,10,10)) #w0,v0, l0,sigmaf,sigman
thetaL =np.array((1e-5,1e-1,1e-1,1e-4,1e-3))
thetaU =np.array((1e2,1e4,1e4,1e2,1e4))


theta0voorgp = np.zeros(2*n_features+3)
thetaLvoorgp = np.zeros(2*n_features+3)
thetaUvoorgp = np.zeros(2*n_features+3)

theta0voorgp[0] = theta0[0] #w0
thetaLvoorgp[0] = thetaL[0] #w0
thetaUvoorgp[0] = thetaU[0] #w0
theta0voorgp[2*n_features+1]= theta0[3] #sigmaf
thetaLvoorgp[2*n_features+1]= thetaL[3] #sigmaf
thetaUvoorgp[2*n_features+1]= thetaU[3] #sigmaf
theta0voorgp[2*n_features + 2] = theta0[4] #sigman
thetaLvoorgp[2*n_features + 2] = thetaL[4] #sigman
thetaUvoorgp[2*n_features + 2] = thetaU[4] #sigman


for i in range(n_features):
    theta0voorgp[1+i]=theta0[1] #v0
    thetaLvoorgp[1+i]=thetaL[1] #v0
    thetaUvoorgp[1+i]=thetaU[1] #v0

    theta0voorgp[n_features+i+1] = theta0[2] #l0
    thetaLvoorgp[n_features+i+1] = thetaL[2] #l0
    thetaUvoorgp[n_features+i+1] = thetaU[2] #l0
              
gp = GaussianProcess(corr='non_stationary',theta0 =theta0voorgp,thetaL=thetaLvoorgp ,thetaU=thetaUvoorgp)
gp.fit(Xtrain, ytrain)
ypred, yolo  = gp.predict(Xvalid, eval_MSE = True)

print yolo
ax = range(1,129,1)

for i in range(yvalid.shape[1]):
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