import numpy as np
from sklearn.gaussian_process import GaussianProcess
import matplotlib.pyplot as plt
from scipy import interpolate

def rchange(arr):
    change = np.zeros(arr.size-1)
    for i in range(change.size):
        change[i] =( arr[i+1]-arr[i])/arr[i]
    return change
    
    
yolo = np.load('y.npy')
tijdquart = np.load('a.npy') #tijdsas voor kwartaal data: - 92 0 enz.
tijdmaand = np.load('b.npy') #tijdsas voor maandelijkse data: - 31 0 28 enz.
qdata = np.load('q.npy') #variable y-as per determinant voor kwartaal data
mdata = np.load('m.npy') #variable y-as per determinant voor maandelijkse data


for j in range(5299,7399,7):
    voorspellingsdag =j # 20 augustus 07, voorspellingsdag modulo 7 moet 0 zijn!!
    
    print "voorspellingsdag is ",j 
    flagq = True
    for i in range(tijdquart.size):
        if flagq:
            if tijdquart[i] >= voorspellingsdag:
                kwartaal_x = tijdquart[0:i]
                flagq = False
    flagm = True
    for i in range(tijdmaand.size):
        if flagm:
            if tijdmaand[i] >= voorspellingsdag:
                maand_x = tijdmaand[0:i]
                flagm = False           
    
    kwartaal_y = qdata[:,0:kwartaal_x.size]
    maand_y = mdata[:,0:maand_x.size]
    
    
    c = np.arange(-7,voorspellingsdag + 7,7) #tijdsas waarvoor cubic spline waarden moet zoeken
    
    x = np.zeros(voorspellingsdag/7+1)
    for i in range(qdata.shape[0]):
        tck = interpolate.splrep(kwartaal_x,kwartaal_y[i],s=0)
        ynew = interpolate.splev(c,tck,der=0)
        #change = rchange(ynew)
        ynew= ynew[1:] #na de rchange hebben we de waarde voor -7 ni meer nodig
        
        x = np.vstack((x, ynew))
        #x = np.vstack((x, np.vstack((ynew,change))))
        
    for i in range(mdata.shape[0]):
        tck = interpolate.splrep(maand_x,maand_y[i],s=0)
        ynew = interpolate.splev(c,tck,der=0)
        #change = rchange(ynew)
        ynew= ynew[1:] #na de rchange hebben we de waarde voor -7 ni meer nodig
        
        x = np.vstack((x, ynew))
        #x = np.vstack((x, np.vstack((ynew,change))))  
        
        
        
    x = x[1:] #eerste rij zeros weg
    x = x.T  
    
    X = x[:voorspellingsdag/7] #alles behalve de laatste waarde (= de waarde van de det op de voorspellingsdag) wordt gebruikt voor training
    y = yolo[:voorspellingsdag/7]
    X = np.hstack((X,X))
    
    n_features = X.shape[1]
    theta0 = np.ones(2*n_features+3)
    thetaL = theta0 * 1e-1
    thetaU = theta0 *2
    gp = GaussianProcess(corr='non_stationary',theta0 =theta0,thetaL=thetaL ,thetaU=thetaU)
    print X.shape, y.shape

    
    gp.fit(X, y)
    '''
    x = x[voorspellingsdag/7:] #de laatste waarde wordt gebruikt om voorspelling te doen 
    pred=gp.predict(x)'''
