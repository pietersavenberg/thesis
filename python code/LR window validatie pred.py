from scipy import interpolate
import numpy as np
from sklearn.linear_model import LinearRegression
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
        return hitrate/(yvalid.shape[0]*yvalid.shape[1])
    except:
        for i in range(yvalid.shape[0]):
            if yvalid[i]*ypred[i] > 0:
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
DSdet = np.load('det.npy') #wekelijkse data vanaf 8 feb 93
tijdquart = np.load('a.npy')#tijdsas voor kwartaal data: - 92 0 tem 7029 = 15 mei 2012
tijdmaand = np.load('b.npy')#tijdsas voor maandelijkse data: - 31 0 28 tem 7029
qdata = np.load('q.npy') #(n_features = 213, n_samples = 79)
mdata = np.load('m.npy') #(n_features = 490, n_samples = 233)

det = 15
n_features = det
    
yvalidlist = np.zeros(15)
ypredlist = np.zeros(15)

n_windows = 4
for i in range(n_windows):
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
    
    for i in range(mdatawindow.shape[0]):
        tck = interpolate.splrep(mtijdwindow,mdatawindow[i],s=0)
        ynew = interpolate.splev(tijdtrainvalid,tck,der=0)
        change = rchange(ynew)
        ynew= ynew[1:] #na de rchange hebben we de waarde voor -7 ni meer nodig
        x = np.vstack((x, np.vstack((ynew,change))))  
        
    for i in range(qdatawindow.shape[0]):
        tck = interpolate.splrep(qtijdwindow,qdatawindow[i],s=0)
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
    

    LR = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
    LR.fit(Xtrain, ytrain) 
    ypred  = LR.predict(Xvalid)
    
    yvalidlist = np.vstack((yvalidlist,yvalid))
    ypredlist = np.vstack((ypredlist,ypred))




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

np.save('ytrain', ytrain)
np.save('ypredlist', ypredlist)
np.save('yvalidlist', yvalidlist)
