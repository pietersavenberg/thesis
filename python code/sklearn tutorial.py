import numpy as np
from matplotlib import pyplot as pl
from sklearn.gaussian_process import GaussianProcess

np.random.seed(1)
def f(x):
    """The function to predict."""
    return np.sin(x)

#----------------------------------------------------------------------
#  First the noiseless case
X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T

# Observations
y = f(X).ravel()

# Mesh the input space for evaluations of the real function, the prediction and its MSE
x = np.atleast_2d(np.linspace(0, 3.14, 1000)).T


# Instanciate a Gaussian Process model
#,thetaL = np.array([1e-4,1e-4,1e-4,1e-4]),thetaU = np.array([1,1,1,1])
gp = GaussianProcess(corr='non_stationary',theta0 = np.array([0.5,0.5,0.5,0.5]),thetaL = np.array([0.1,1e-1,1e-1,1e-1]),thetaU = np.array([2,2,2,1]),random_start=2)
#gp = GaussianProcess(corr='squared_exponential',theta0 = np.array([1e-1]),thetaL = np.array([1e-4]),thetaU = np.array([1]),random_start=10)

'''class sklearn.gaussian_process.GaussianProcess(regr='constant', corr='squared_exponential', 
beta0=None, storage_mode='full', verbose=False, theta0=0.1, thetaL=None, thetaU=None, 
optimizer='fmin_cobyla', random_start=1, normalize=True, nugget=2.2204460492503131e-15, random_state=None)

http://scikit-learn.org/0.13/modules/generated/sklearn.gaussian_process.GaussianProcess.html
'''
# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y= gp.predict(x)
print y
#sigma = np.sqrt(MSE)



'''
# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
fig = pl.figure()
pl.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
pl.plot(X, y, 'r.', markersize=10, label=u'Observations')
pl.plot(x, y_pred, 'b-', label=u'Prediction')
pl.fill(np.concatenate([x, x[::-1]]),
        np.concatenate([y_pred - 1.9600 * sigma,
                       (y_pred + 1.9600 * sigma)[::-1]]),
        alpha=.5, fc='b', ec='None', label='95% confidence interval')
pl.xlabel('$x$')
pl.ylabel('$f(x)$')
pl.ylim(-10, 20)
pl.legend(loc='upper left')

#----------------------------------------------------------------------
# now the noisy case
X = np.linspace(0.1, 9.9, 20)
X = np.atleast_2d(X).T

# Observations and noise
y = f(X).ravel()
dy = 0.5 + 1.0 * np.random.random(y.shape)
noise = np.random.normal(0, dy)
y += noise

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0, 10, 1000)).T

# Instanciate a Gaussian Process model
gp = GaussianProcess(corr='squared_exponential', theta0=1e-1,
                     thetaL=1e-3, thetaU=1,
                     nugget=(dy / y) ** 2,
                     random_start=100)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, MSE = gp.predict(x, eval_MSE=True)
sigma = np.sqrt(MSE)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
fig = pl.figure()
pl.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
pl.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label=u'Observations')
pl.plot(x, y_pred, 'b-', label=u'Prediction')
pl.fill(np.concatenate([x, x[::-1]]),
        np.concatenate([y_pred - 1.9600 * sigma,
                       (y_pred + 1.9600 * sigma)[::-1]]),
        alpha=.5, fc='b', ec='None', label='95% confidence interval')
pl.xlabel('$x$')
pl.ylabel('$f(x)$')
pl.ylim(-10, 20)
pl.legend(loc='upper left')

pl.show()
'''
