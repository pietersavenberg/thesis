import Quandl
import numpy as np
from pulp import *
from pandas import *
from sklearn.gaussian_process import GaussianProcess

'''
start = "November 2001"
end = "November 2012"
freq = "monthly"
BELlag = np.array(Quandl.get("YAHOO/INDEX_BFX",collapse=freq,trim_start = start, trim_end= end,transformation = "rdiff")["Close"])
print BELlag.shape
x = LpVariable("x", 0, 3)

'''
def squared_exponential(theta, d):
    """
    Squared exponential correlation model (Radial Basis Function).
    (Infinitely differentiable stochastic process, very smooth)::

                                            n
        theta, dx --> r(theta, dx) = exp(  sum  - theta_i * (dx_i)^2 )
                                          i = 1

    Parameters
    ----------
    theta : array_like
        An array with shape 1 (isotropic) or n (anisotropic) giving the
        autocorrelation parameter(s).

    dx : array_like
        An array with shape (n_eval, n_features) giving the componentwise
        distances between locations x and x' at which the correlation model
        should be evaluated.

    Returns
    -------
    r : array_like
        An array with shape (n_eval, ) containing the values of the
        autocorrelation model.
    """

    theta = np.asarray(theta, dtype=np.float)
    d = np.asarray(d, dtype=np.float)

    if d.ndim > 1:
        n_features = d.shape[1]
    else:
        n_features = 1

    if theta.size == 1:
        return np.exp(-theta[0] * np.sum(d ** 2, axis=1))
    elif theta.size != n_features:
        raise ValueError("Length of theta must be 1 or %s" % n_features)
    else:
        return np.exp(-np.sum(theta.reshape(1, n_features) * d ** 2, axis=1))
        
def linear(x):
    """
    First order polynomial (linear, p = n+1) regression model.

    x --> f(x) = [ 1, x_1, ..., x_n ].T

    Parameters
    ----------
    x : array_like
        An array with shape (n_eval, n_features) giving the locations x at
        which the regression model should be evaluated.

    Returns
    -------
    f : array_like
        An array with shape (n_eval, p) with the values of the regression
        model.
    """
    x = np.asarray(x, dtype=np.float)
    n_eval = x.shape[0]
    f = np.hstack([np.ones([n_eval, 1]), x])
    return f
    
def quadratic(x):
    """
    Second order polynomial (quadratic, p = n*(n-1)/2+n+1) regression model.

    x --> f(x) = [ 1, { x_i, i = 1,...,n }, { x_i * x_j,  (i,j) = 1,...,n } ].T
                                                          i > j

    Parameters
    ----------
    x : array_like
        An array with shape (n_eval, n_features) giving the locations x at
        which the regression model should be evaluated.

    Returns
    -------
    f : array_like
        An array with shape (n_eval, p) with the values of the regression
        model.
    """

    x = np.asarray(x, dtype=np.float)
    n_eval, n_features = x.shape
    f = np.hstack([np.ones([n_eval, 1]), x])
    for k in range(n_features):
        f = np.hstack([f, x[:, k, np.newaxis] * x[:, k:]])

    return f


def manhattan_distances(X, Y=None, sum_over_features=True,
                        size_threshold=5e8):
    """ Compute the L1 distances between the vectors in X and Y.

    With sum_over_features equal to False it returns the componentwise
    distances.

    Parameters
    ----------
    X : array_like
        An array with shape (n_samples_X, n_features).

    Y : array_like, optional
        An array with shape (n_samples_Y, n_features).

    sum_over_features : bool, default=True
        If True the function returns the pairwise distance matrix
        else it returns the componentwise L1 pairwise-distances.

    size_threshold : int, default=5e8
        Avoid creating temporary matrices bigger than size_threshold (in
        bytes). If the problem size gets too big, the implementation then
        breaks it down in smaller problems.

    Returns
    -------
    D : array
        If sum_over_features is False shape is
        (n_samples_X * n_samples_Y, n_features) and D contains the
        componentwise L1 pairwise-distances (ie. absolute difference),
        else shape is (n_samples_X, n_samples_Y) and D contains
        the pairwise l1 distances.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import manhattan_distances
    >>> manhattan_distances(3, 3)#doctest:+ELLIPSIS
    array([[ 0.]])
    >>> manhattan_distances(3, 2)#doctest:+ELLIPSIS
    array([[ 1.]])
    >>> manhattan_distances(2, 3)#doctest:+ELLIPSIS
    array([[ 1.]])
    >>> manhattan_distances([[1, 2], [3, 4]],\
         [[1, 2], [0, 3]])#doctest:+ELLIPSIS
    array([[ 0.,  2.],
           [ 4.,  4.]])
    >>> import numpy as np
    >>> X = np.ones((1, 2))
    >>> y = 2 * np.ones((2, 2))
    >>> manhattan_distances(X, y, sum_over_features=False)#doctest:+ELLIPSIS
    array([[ 1.,  1.],
           [ 1.,  1.]]...)
    """
 
    temporary_size = X.size * Y.shape[-1]
    # Convert to bytes
    temporary_size *= X.itemsize
    if temporary_size > size_threshold and sum_over_features:
        # Broadcasting the full thing would be too big: it's on the order
        # of magnitude of the gigabyte
        D = np.empty((X.shape[0], Y.shape[0]), dtype=X.dtype)
        index = 0
        increment = 1 + int(size_threshold / float(temporary_size) *
                            X.shape[0])
        while index < X.shape[0]:
            this_slice = slice(index, index + increment)
            tmp = X[this_slice, np.newaxis, :] - Y[np.newaxis, :, :]
            tmp = np.abs(tmp, tmp)
            tmp = np.sum(tmp, axis=2)
            D[this_slice] = tmp
            index += increment
    else:
        D = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
        D = np.abs(D, D)
        if sum_over_features:
            D = np.sum(D, axis=2)
        else:
            D = D.reshape((-1, X.shape[1]))
    return D    

a = [[1,2,3],[4,5,6]] #input waarvoor moet voorspeld worden: 2 datapunten, 3 features
a = np.array(a)
b = [[1,2,4],[4,5,6],[7,8,9],[10,11,12]] #trainingsset: 4 datapunten, 3 features
b = np.array(b)
d = manhattan_distances(a,b,sum_over_features=False)


n_features = 3
theta = np.array([5,6,7])
print d
#print np.exp(-0.5*np.sum((theta.reshape(1, n_features) * d) ** 2,axis=1)).reshape(2,3)
print np.sum(theta.reshape(1, n_features)*d**2,axis = 1)+5 #8 combinaties van datapunten, de features worden gebundeld per waarde
#print quadratic(a)



'''
gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1,
                     random_start=10)
y = np.array([1,4,9])
gp.fit(b, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
a,b,c,d = gp.predict(a, eval_MSE=True)
print a
'''