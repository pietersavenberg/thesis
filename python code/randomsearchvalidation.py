import numpy as np
from sklearn.gaussian_process import GaussianProcess
from sklearn.metrics import *
import matplotlib.pyplot as plt
from matplotlib.colors import *
from mpl_toolkits.mplot3d import Axes3D
#import mpl_toolkits.mplot3d 
from scipy import interpolate

det = 30 #aantal determinanten die we overhouden (computatie sneller)

#eerst train/validatie scheiden
y = np.load('y.npy')
yalsdet = np.load('yalsdet.npy')
tijdquart = np.load('a.npy')#tijdsas voor kwartaal data: - 92 0 tem 7029 = 15 mei 2012
tijdmaand = np.load('b.npy')#tijdsas voor maandelijkse data: - 31 0 "28 tem 7029
qdata = np.load('q.npy') #variable y-as per determinant voor kwartaal data
mdata = np.load('m.npy') #variable y-as per determinant voor maandelijkse data

tijdtrainvalid = range(0,7035,7)#tijdsas waarvoor cubic spline waarden moet zoeken, loopt tem 7028 = 14 mei 2012

todelete = []
tijdvalid = []
yvalid = []
for i in range(1,17):
    todelete += range(60*i,60*i+8)
    tijdvalid += tijdtrainvalid[60*i:60*i+8]
    yvalid += list(y[60*i:60*i+8])
tijdtrainvalid = np.asarray(tijdtrainvalid)   
tijdvalid =  np.asarray(tijdvalid)   
tijdtrain = np.delete(tijdtrainvalid,todelete)
ytrain = np.delete(y,todelete,axis = 0)

yvalid = np.asarray(yvalid)


x = np.zeros(tijdtrain.size)
for i in range(qdata.shape[0]):
    tck = interpolate.splrep(tijdquart,qdata[i],s=0)
    ynew = interpolate.splev(tijdtrain,tck,der=0)
    #change = rchange(ynew)
    #ynew= ynew[1:] #na de rchange hebben we de waarde voor -7 ni meer nodig
    
    x = np.vstack((x, ynew))
    #x = np.vstack((x, np.vstack((ynew,change))))

for i in range(mdata.shape[0]):
    tck = interpolate.splrep(tijdmaand,mdata[i],s=0)
    ynew = interpolate.splev(tijdtrain,tck,der=0)
    #change = rchange(ynew)
    #ynew= ynew[1:] #na de rchange hebben we de waarde voor -7 ni meer nodig
    
    x = np.vstack((x, ynew))
    #x = np.vstack((x, np.vstack((ynew,change))))  
    
#yalsdet = yalsdet[:,:1006]
#x = np.vstack((x,yalsdet))

x = x[1:] #eerste rij zeros weg
x = x.T 


Xtrain = x[:,:det] #voor snellere computatie, 30 determinanten ipv 700
n_features = Xtrain.shape[1]

x = np.zeros(tijdvalid.size)
for i in range(qdata.shape[0]):
    tck = interpolate.splrep(tijdquart,qdata[i],s=0)
    ynew = interpolate.splev(tijdvalid,tck,der=0)
    #change = rchange(ynew)
    #ynew= ynew[1:] #na de rchange hebben we de waarde voor -7 ni meer nodig
    
    x = np.vstack((x, ynew))
    #x = np.vstack((x, np.vstack((ynew,change))))

for i in range(mdata.shape[0]):
    tck = interpolate.splrep(tijdmaand,mdata[i],s=0)
    ynew = interpolate.splev(tijdvalid,tck,der=0)
    #change = rchange(ynew)
    #ynew= ynew[1:] #na de rchange hebben we de waarde voor -7 ni meer nodig
    
    x = np.vstack((x, ynew))
    #x = np.vstack((x, np.vstack((ynew,change)))) 

x = x[1:] #eerste rij zeros weg
x = x.T 


Xvalid = x[:,:det] #voor snellere computatie, 30 determinanten ipv 700


#nu random log search voor theta, dan gp = (geen waarden voor thetaU en L geven!), fit en predict en dan score berekenen
theta0 = np.ones(5) #w0,v0, l0,sigmaf,sigman
thetaL = theta0*1e-4
thetaU = theta0*1e5

#plot lijsten aanmaken:
xaxis,yaxis,zaxis,yolo = [],[],[],[]
for i in range(100):
    log10theta0 = np.log10(thetaL) + np.random.rand(theta0.size).reshape(theta0.shape)* np.log10(thetaU / thetaL)
    theta0 = 10. ** log10theta0
    #print theta0
    theta0voorgp = np.ones(2*n_features+3)
    
    
    theta0voorgp[0] = theta0[0] #w0
    #theta0voorgp[0] = 1#w0
    
    #theta0voorgp[2*n_features+1]= theta0[3] #sigmaf
    theta0voorgp[2*n_features+1]= 1 #sigmaf
    
    #theta0voorgp[2*n_features + 2] = theta0[4] #sigman
    theta0voorgp[2*n_features + 2] = 1 #sigman
    
    for i in range(n_features):
        #theta0voorgp[1+i]=theta0[1] #v0
        theta0voorgp[n_features+i+1] = theta0[2] #l0
        theta0voorgp[1+i]=1 #v0
        #theta0voorgp[n_features+i+1] = 1#l0
    gp = GaussianProcess(corr='non_stationary',theta0 =theta0voorgp,thetaL=None ,thetaU=None)
    #print Xtrain.shape,ytrain.shape
    gp.fit(Xtrain, ytrain) 
    ypred  = gp.predict(Xvalid)
    
    
    #use sklearn score function voor ypred vs. yvalid
    
    z= mean_squared_error(yvalid, ypred)  
    #info voor plot
    xaxis.append(theta0[0]) #sigman, de andere drie zijn 1
    yaxis.append(theta0[2]) #sigmaf
    #yolo.append(theta0[4]) #sigman
    zaxis.append(z)


cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(xaxis, yaxis,s=50, c=zaxis, cmap=cm,norm=LogNorm())
plt.yscale('log')
plt.xscale('log')
plt.xlim(1e-5,1e5)
plt.ylim(1e-5,1e5)
plt.xlabel('w_0')
plt.ylabel('l_0')
plt.colorbar(sc)
plt.show()





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
