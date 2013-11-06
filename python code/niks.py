import numpy as np
from scipy import rand
a=1
b=2
c = np.hstack([a,b])
d = np.array([1,2,3,4,5,6,7,8,9])
e = d[1:4]




l0=1e-1
lL=None
lU=None
w0=None
wL=None
wU = None
v0=None
vL = None
vU = None
sigmaf =1e-1
sigmaL = None
sigmaU = None

thetaL = np.hstack([wL,vL,lL,sigmaL])

n_features = 1
theta = np.array([1,1,1,4])
theta = np.asarray(theta, dtype=np.float)


