import numpy as np
from sklearn.gaussian_process import GaussianProcess
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
                

y = np.load('y.npy')
yalsdet = np.load('yalsdet.npy')
DSdet = np.load('det.npy') #wekelijkse data vanaf 8 feb 93
tijdquart = np.load('a.npy')#tijdsas voor kwartaal data: - 92 0 tem 7029 = 15 mei 2012
tijdmaand = np.load('b.npy')#tijdsas voor maandelijkse data: - 31 0 28 tem 7029
qdata = np.load('q.npy') #(n_features = 213, n_samples = 79)
mdata = np.load('m.npy') #(n_features = 490, n_samples = 233)

det = 99
n_features = det


theta0 = np.ones(5) #w0,v0, l0,sigmaf,sigman
lower,upper = 1e-4,1e4
thetaL = theta0*lower
thetaU = theta0*upper

#thetaL =np.array((1e-5,1e-1,1e-1,1e-4,1e-3))
#thetaU =np.array((1e2,1e4,1e4,1e2,1e2))

thetaL =np.array((1e-5,1e-2,1e-2,1e-2,1e-2))
thetaU =np.array((1e0,1e4,1e4,1e1,1e1))

n_windows = 1 #aantal windows
for i in range(n_windows):
    i = i+19
    qdatawindow = qdata[:,i:42+i] 
    mdatawindow = mdata[:,3*i:122+3*i]
    
    qtijdwindow = tijdquart[i:42+i]
    mtijdwindow = tijdmaand[3*i:122+3*i]
    
    DSwindow = DSdet[:,13*i:536+13*i] #8 feb 93 tem 12 mei: train + validatie dus 
    yalsdetwindow = yalsdet[13*i:535+13*i]
    ywindow = y[13*i:535+13*i]
    
    tijdtrainvalid = range(-7+13*7*i,3745+13*7*i,7)#tijdsas waarvoor cubic spline waarden moet zoeken voor de determinanten
    
    x = np.zeros(len(tijdtrainvalid)-1)
    for i in range(DSwindow.shape[0]):
        changedet = rchange(DSwindow[i])
        x = np.vstack((x, np.vstack((DSwindow[i,1:],changedet))))
    
    for i in range(qdatawindow.shape[0]):
        tck = interpolate.splrep(qtijdwindow,qdatawindow[i],s=0)
        ynew = interpolate.splev(tijdtrainvalid,tck,der=0)
        change = rchange(ynew)
        ynew= ynew[1:] #na de rchange hebben we de waarde voor -7 ni meer nodig
        x = np.vstack((x, np.vstack((ynew,change))))
    
    for i in range(mdatawindow.shape[0]):
        tck = interpolate.splrep(mtijdwindow,mdatawindow[i],s=0)
        ynew = interpolate.splev(tijdtrainvalid,tck,der=0)
        change = rchange(ynew)
        ynew= ynew[1:] #na de rchange hebben we de waarde voor -7 ni meer nodig
        x = np.vstack((x, np.vstack((ynew,change))))  
    
    
    x = x[1:] #remove first row zeros
    x = x.T
    x = np.hstack((yalsdetwindow,x))

    
    Xtrain = x[:522]
    Xvalid = x[522:]
    
    ytrain = ywindow[:522]
    yvalid = ywindow[522:]
    
    Xvalid = Xvalid[:,:det]
    Xtrain = Xtrain[:,:det]
    
    #plot lijsten aanmaken voor colorplots:
    xaxis,yaxis,zaxis,hit= [],[],[],[]
    for i in range(50):
        log10theta0 = np.log10(thetaL) + np.random.rand(theta0.size).reshape(theta0.shape)* np.log10(thetaU / thetaL)
        theta0 = 10. ** log10theta0
        print theta0
        theta0voorgp = np.zeros(2*n_features+3)
        
        theta0voorgp[0] = theta0[0] #w0
        #theta0voorgp[0] = 1#w0
        
        theta0voorgp[2*n_features+1]= theta0[3] #sigmaf
        #theta0voorgp[2*n_features+1]= 1e3#sigmaf
        
        theta0voorgp[2*n_features + 2] = theta0[4] #sigman
        #theta0voorgp[2*n_features + 2] = 1e3 #sigman
        
        for i in range(n_features):
            theta0voorgp[1+i]=theta0[1] #v0
            #theta0voorgp[1+i]=1 #v0
            
            theta0voorgp[n_features+i+1] = theta0[2] #l0
            #theta0voorgp[n_features+i+1] = 1#l0
        gp = GaussianProcess(corr='non_stationary',verbose = False ,theta0 =theta0voorgp,thetaL=None ,thetaU=None)
        gp.fit(Xtrain, ytrain) 
        ypred  = gp.predict(Xvalid)
        
        
        #use sklearn score function voor ypred vs. yvalid
        
        z= score(yvalid, ypred)  
        #info voor plot
        xaxis.append(theta0[1]) #sigman, de andere drie zijn 1
        yaxis.append(theta0[2]) #sigmaf
        zaxis.append(z)
    
        if z>0.191:
            print "theta0 is: ", theta0
            print "bijhorende score is ", z
            print "NMSE ",NMSE(yvalid,ypred)
            print "hitrate ",hitrate(yvalid,ypred)
    


    '''
    cm = plt.cm.get_cmap('RdYlBu')
    sc = plt.scatter(xaxis, yaxis,s=50, c=zaxis, cmap=cm,norm=LogNorm())
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(lower,upper)
    plt.ylim(lower,upper)
    plt.xlabel('v_0')
    plt.ylabel('l_0')
    plt.colorbar(sc)
    plt.show()
    '''
