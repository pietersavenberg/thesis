import numpy as np
from sklearn.gaussian_process import GaussianProcess
import matplotlib.pyplot as plt
from scipy import interpolate

def rchange(arr):
    change = np.zeros(arr.size-1)
    for i in range(change.size):
        change[i] =( arr[i+1]-arr[i])/arr[i]
    return change
    
    
y = np.load('y.npy')
a = np.load('a.npy') #vaste x-as voor kwartaal data: - 92 0 enz.
b = np.load('b.npy') #variable y-as per determinant voor kwartaal data

voorspellingsdag =5299 # 20 augustus 07, voorspellingsdag modulo 7 moet 0 zijn!!
flag = True
for i in range(a.size):
    if flag:
        if a[i] >= voorspellingsdag:
            kwartaal_x = a[0:i]
            flag = False
            

kwartaal_y = b[:,0:kwartaal_x.size]
c = np.arange(-7,voorspellingsdag + 7,7)

x = np.zeros(voorspellingsdag/7+1)
for i in range(b.shape[0]):
    tck = interpolate.splrep(kwartaal_x,kwartaal_y[i],s=0)
    ynew = interpolate.splev(c,tck,der=0)
    change = rchange(ynew)
    ynew= ynew[1:] #na de rchange hebben we de waarde voor -7 ni meer nodig
    
    x = np.vstack((x, np.vstack((ynew,change))))
x = x[1:] #eerste rij zeros weg
x = x.T  

X = x[:voorspellingsdag/7] #alles behalve de laatste waarde (= de waarde van de det op de voorspellingsdag) wordt gebruikt voor training
y = y[:voorspellingsdag/7]


n_features = X.shape[1]
theta0 = np.ones(2*n_features+2)
thetaL = theta0 * 1e-1
thetaU = theta0 *2
gp = GaussianProcess(corr='non_stationary',theta0 =theta0,thetaL=thetaL ,thetaU=thetaU)

print X.shape,y.shape, kwartaal_x.shape

'''
gp.fit(X, y)

x = x[voorspellingsdag/7:] #de laatste waarde wordt gebruikt om voorspelling te doen 
pred=gp.predict(x)
#pred = gp.predict(
'''