from scipy import interpolate
import numpy as np
from sklearn.gaussian_process import GaussianProcess
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
DSdet = np.load('det.npy') #wekelijkse data vanaf 8 feb 93
tijdquart = np.load('a.npy')#tijdsas voor kwartaal data: - 92 0 tem 7029 = 15 mei 2012
tijdmaand = np.load('b.npy')#tijdsas voor maandelijkse data: - 31 0 28 tem 7029
qdata = np.load('q.npy') #(n_features = 213, n_samples = 79)
mdata = np.load('m.npy') #(n_features = 490, n_samples = 233)

det = 1079
n_features = det
n_windows = 4

theta0 = np.array((1e-5,100,100,1,1)) #w0,v0, l0,sigmaf,sigman

thetaL =np.array((1e-6,1e-1,1e-1,1e-2,1e-2))
thetaU =np.array((1e0,1e6,1e6,1e1,1e1))

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
    
yvalidlist = np.zeros(15)
ypredlist = np.zeros(15)
sigmalist = np.zeros(15)

#allereerste punt van de eerste validatiewindow set is x = 17 feb 03, y = 24 feb 03. Dus eerste voorspelling van y gebeurt voor waarde van y op 24 feb. 
#Eerste allocatie gebeurt op 17 feb

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
    

    gp = GaussianProcess(corr='non_stationary',theta0 =theta0voorgp,thetaL=thetaLvoorgp ,thetaU=thetaUvoorgp)
    #gp = GaussianProcess(corr='non_stationary',theta0 =theta0voorgp,thetaL=None ,thetaU=None)

    gp.fit(Xtrain, ytrain)
    ypred,MSE ,thetaopt= gp.predict(Xvalid,eval_MSE = True)
    
    yvalidlist = np.vstack((yvalidlist,yvalid))
    ypredlist = np.vstack((ypredlist,ypred))
    MSE = MSE.T
    sigma = np.sqrt(MSE)
    sigmalist = np.vstack((sigmalist,sigma))



yvalidlist = yvalidlist[1:] #remove zeros
ypredlist = ypredlist[1:]
sigmalist = sigmalist[1:]

ax = range(522,yvalidlist.shape[0]+522,1)


for i in range(yvalidlist.shape[1]):
    plt.figure(i+1)
    plt.fill_between(ax, ypredlist[:,i]-1.96*(sigmalist[:,i]), ypredlist[:,i]+1.96*(sigmalist[:,i]),alpha = .5,  facecolor='blue', interpolate=True,label='95% confidence interval')
    plt.plot(ax, ypredlist[:,i],'b-',label=u'Predictions')
    plt.plot(ax, yvalidlist[:,i],'r-',label=u'Observations')
    plt.xlabel('time (weeks)')
    plt.ylabel('return')
    plt.vlines(range(12+522,len(ax)+522,13),-0.1,0.1,'y')
    plt.title(('Country',i))
    plt.legend(loc='upper right')
    fig = plt.gcf()
    fig.set_size_inches(15,8)


astheta = range(det)
plt.figure(20)
plt.plot(astheta,thetaopt[1:det+1],'r.',label=u'l')
plt.plot(astheta,thetaopt[det+1:2*det+1],'b.',label=u'v')
plt.yscale('log')
plt.title("Lengteschalen")
plt.legend(loc='upper right')




plt.show()

for i in range(15):
    print "Country ",i," NMSE :", NMSE(yvalidlist[:,i],ypredlist[:,i])
    print "Country ",i," hitrate :", hitrate(yvalidlist[:,i],ypredlist[:,i])

print "mean NMSE :", NMSE(yvalidlist,ypredlist)/(yvalidlist.shape[1])
print "mean hitrate :", hitrate(yvalidlist,ypredlist)

np.save('ytrain', ytrain)
np.save('ypredlist', ypredlist)
np.save('yvalidlist', yvalidlist)
np.save('sigmalist', sigmalist)
