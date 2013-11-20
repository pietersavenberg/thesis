import numpy as np
from sklearn.gaussian_process import GaussianProcess
import matplotlib.pyplot as plt
from scipy import interpolate

#eerst train/validatie scheiden
y = np.load('y.npy')
yalsdet = np.load('yalsdet.npy')
tijdquart = np.load('a.npy')#tijdsas voor kwartaal data: - 92 0 tem 7029 = 15 mei 2012
tijdmaand = np.load('b.npy')#tijdsas voor maandelijkse data: - 31 0 28 tem 7029
qdata = np.load('q.npy') #variable y-as per determinant voor kwartaal data
mdata = np.load('m.npy') #variable y-as per determinant voor maandelijkse data

tijdtrainvalid = range(0,7035,7)#tijdsas waarvoor cubic spline waarden moet zoeken, loopt tem 7028 = 14 mei 2012

todelete = []
tijdvalid = []
for i in range(1,17):
    todelete += range(60*i,60*i+8)
    tijdvalid += tijdtrainvalid[60*i:60*i+8]
    
tijdtrainvalid = np.asarray(tijdtrainvalid)   
tijdvalid =  np.asarray(tijdvalid)   
tijdtrain = np.delete(tijdtrainvalid,todelete)
ytrain = np.delete(y,todelete,axis = 0)


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

#nu random log search voor theta, dan gp = (geen waarden voor thetaU en L geven!), fit en predict en dan score berekenen
