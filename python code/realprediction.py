import numpy as np
from sklearn.gaussian_process import GaussianProcess

y = np.load('y.npy')
x = np.load('x.npy')
n_features = x.shape[1]
theta0 = np.ones(2*n_features+2)
thetaL = theta0 * 1e-1
thetaU = theta0 *2
gp = GaussianProcess(corr='non_stationary',theta0 =theta0,thetaL=thetaL ,thetaU=thetaU)

gp.fit(x, y)
