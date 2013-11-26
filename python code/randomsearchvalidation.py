import numpy as np
from sklearn.gaussian_process import GaussianProcess
from sklearn.metrics import *
import matplotlib.pyplot as plt
from matplotlib.colors import *
from mpl_toolkits.mplot3d import Axes3D
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
                
det = 99 #aantal determinanten die we overhouden (computatie sneller), 15 van yalsdet + 42*2 van DSdet

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
tijdvalid =np.asarray(tijdvalid)

#print Xtrain.shape, ytrain.shape, Xvalid.shape
Xvalid = Xvalid[:,:det]
Xtrain = Xtrain[:,:det]

#nu random log search voor theta, dan gp = (geen waarden voor thetaU en L geven!), fit en predict en dan score berekenen
theta0 = np.ones(5) #w0,v0, l0,sigmaf,sigman
lower,upper = 1e-4,1e5
thetaL = theta0*lower
thetaU = theta0*upper

n_features = Xtrain.shape[1]
'''
#plot lijsten aanmaken:
xaxis,yaxis,zaxis,hit= [],[],[],[]
for i in range(100):
    log10theta0 = np.log10(thetaL) + np.random.rand(theta0.size).reshape(theta0.shape)* np.log10(thetaU / thetaL)
    theta0 = 10. ** log10theta0
    theta0voorgp = np.zeros(2*n_features+3)
    
    #theta0voorgp[0] = theta0[0] #w0
    theta0voorgp[0] = 1#w0
    
    #theta0voorgp[2*n_features+1]= theta0[3] #sigmaf
    theta0voorgp[2*n_features+1]= 1e3#sigmaf
    
    theta0voorgp[2*n_features + 2] = theta0[4] #sigman
    #theta0voorgp[2*n_features + 2] = 1e3 #sigman
    
    for i in range(n_features):
        theta0voorgp[1+i]=theta0[1] #v0
        #theta0voorgp[1+i]=1 #v0
        
        #theta0voorgp[n_features+i+1] = theta0[2] #l0
        theta0voorgp[n_features+i+1] = 1#l0
    gp = GaussianProcess(corr='non_stationary',theta0 =theta0voorgp,thetaL=None ,thetaU=None)
    gp.fit(Xtrain, ytrain) 
    ypred  = gp.predict(Xvalid)
    
    
    #use sklearn score function voor ypred vs. yvalid
    
    z= NMSE(yvalid, ypred)  
    #info voor plot
    xaxis.append(theta0[1]) #sigman, de andere drie zijn 1
    yaxis.append(theta0[4]) #sigmaf
    zaxis.append(z)
    hit.append(hitrate(yvalid,ypred))


cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(xaxis, yaxis,s=50, c=zaxis, cmap=cm,norm=LogNorm())
plt.yscale('log')
plt.xscale('log')
plt.xlim(lower,upper)
plt.ylim(lower,upper)
plt.xlabel('v_0')
plt.ylabel('sigma_n')
plt.colorbar(sc)
plt.show()

'''


theta0 = np.array((1e-3,10,1,10,10)) #w0,v0, l0,sigmaf,sigman
thetaL =np.array((1e-5,1e-2,1e-2,1e-2,1e-2))
thetaU =np.array((1e1,1e4,1e4,1e3,1e3))

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
ypred  = gp.predict(Xvalid)


ax = range(0,128,1)

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


'''
fig = plt.figure()
cm = plt.cm.get_cmap('RdYlBu')
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(xaxis, yaxis, yolo,s = 50, c=zaxis,cmap=cm,norm=LogNorm())

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
fig.colorbar(p)
plt.show()
'''
