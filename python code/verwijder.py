import numpy as np
from sklearn.utils import array2d
from sklearn.gaussian_process import GaussianProcess
from sklearn.utils import *

def check_pairwise_arrays(X, Y):
    """ Set X and Y appropriately and checks inputs

    If Y is None, it is set as a pointer to X (i.e. not a copy).
    If Y is given, this does not happen.
    All distance metrics should use this function first to assert that the
    given parameters are correct and safe to use.

    Specifically, this function first ensures that both X and Y are arrays,
    then checks that they are at least two dimensional while ensuring that
    their elements are floats. Finally, the function checks that the size
    of the second dimension of the two arrays is equal.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples_a, n_features]

    Y : {array-like, sparse matrix}, shape = [n_samples_b, n_features]

    Returns
    -------
    safe_X : {array-like, sparse matrix}, shape = [n_samples_a, n_features]
        An array equal to X, guaranteed to be a numpy array.

    safe_Y : {array-like, sparse matrix}, shape = [n_samples_b, n_features]
        An array equal to Y if Y was not None, guaranteed to be a numpy array.
        If Y was None, safe_Y will be a pointer to X.

    """
    if Y is X or Y is None:
        X = Y = atleast2d_or_csr(X)
    else:
        X = atleast2d_or_csr(X)
        Y = atleast2d_or_csr(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices: "
                         "X.shape[1] == %d while Y.shape[1] == %d" % (
                             X.shape[1], Y.shape[1]))

    if not (X.dtype == Y.dtype == np.float32):
        if Y is X:
            X = Y = safe_asarray(X, dtype=np.float)
        else:
            X = safe_asarray(X, dtype=np.float)
            Y = safe_asarray(Y, dtype=np.float)
    return X, Y
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

def manhattan_multiply(X, Y=None, sum_over_features=True,
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
    X, Y = check_pairwise_arrays(X, Y)

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
        D = X[:, np.newaxis, :] * Y[np.newaxis, :, :]
        D = np.abs(D, D)
        if sum_over_features:
            D = np.sum(D, axis=2)
        else:
            D = D.reshape((-1, X.shape[1]))
    return D
    
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
    X, Y = check_pairwise_arrays(X, Y)

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
    
def l1_cross_distances(X):
    """
    Computes the nonzero componentwise L1 cross-distances between the vectors
    in X.

    Parameters
    ----------

    X: array_like
        An array with shape (n_samples, n_features)

    Returns
    -------

    D: array with shape (n_samples * (n_samples - 1) / 2, n_features)
        The array of componentwise L1 cross-distances.

    ij: arrays with shape (n_samples * (n_samples - 1) / 2, 2)
        The indices i and j of the vectors in X associated to the cross-
        distances in D: D[k] = np.abs(X[ij[k, 0]] - Y[ij[k, 1]]).
    """
    X = array2d(X)
    n_samples, n_features = X.shape
    n_nonzero_cross_dist = n_samples * (n_samples - 1) / 2
    ij = np.zeros((n_nonzero_cross_dist, 2), dtype=np.int)
    D = np.zeros((n_nonzero_cross_dist, n_features))
    ll_1 = 0
    for k in range(n_samples - 1):
        ll_0 = ll_1
        ll_1 = ll_0 + n_samples - k - 1
        ij[ll_0:ll_1, 0] = k
        ij[ll_0:ll_1, 1] = np.arange(k + 1, n_samples)
        D[ll_0:ll_1] = np.abs(X[k] - X[(k + 1):n_samples])

    return D, ij.astype(np.int)

def l1_multiply(X):
    """
    Computes the nonzero componentwise L1 cross-distances between the vectors
    in X.

    Parameters
    ----------

    X: array_like
        An array with shape (n_samples, n_features)

    Returns
    -------

    D: array with shape (n_samples * (n_samples - 1) / 2, n_features)
        The array of componentwise L1 cross-distances.

    ij: arrays with shape (n_samples * (n_samples - 1) / 2, 2)
        The indices i and j of the vectors in X associated to the cross-
        distances in D: D[k] = np.abs(X[ij[k, 0]] - Y[ij[k, 1]]).
    """
    X = array2d(X)
    n_samples, n_features = X.shape
    n_nonzero_cross_dist = n_samples * (n_samples - 1) / 2
    ij = np.zeros((n_nonzero_cross_dist, 2), dtype=np.int)
    D = np.zeros((n_nonzero_cross_dist, n_features))
    ll_1 = 0
    for k in range(n_samples - 1):
        ll_0 = ll_1
        ll_1 = ll_0 + n_samples - k - 1
        ij[ll_0:ll_1, 0] = k
        ij[ll_0:ll_1, 1] = np.arange(k + 1, n_samples)
        D[ll_0:ll_1] = np.abs(X[k] * X[(k + 1):n_samples])

    return D, ij.astype(np.int)
 

a = [[1,2,3],[4,5,6]] #input waarvoor moet voorspeld worden: 2 datapunten, 3 features
a = np.array(a)
b = [[1,2,4],[4,5,6],[7,8,9],[10,11,12]] #trainingsset: 4 datapunten, 3 features
b = np.array(b)
d = manhattan_distances(a,b,sum_over_features=False)

D = a[:, np.newaxis, :] * b[np.newaxis, :, :]
D = np.abs(D, D)
D = D.reshape((-1, a.shape[1]))
            


X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
def f(x):
    return x * np.sin(x)
# Observations
y = f(X).ravel()

# Mesh the input space for evaluations of the real function, the prediction and its MSE
x = np.atleast_2d(np.linspace(0, 10, 1000)).T

d =  manhattan_distances(x,X,sum_over_features=False) 
#print d.shape
n_features = 1
theta = np.array([1e-1,1e-1,1e-1,1e-1])
#print np.exp(-0.5*np.sum((theta.reshape(1, n_features) * d) ** 2,axis=1)).reshape(2,3)
w0=theta[0]
v0=theta[1:(n_features+1)]
l0=theta[(n_features+1):(2*n_features+1)]
sigmaf=theta[(2*n_features+1)]

multiplyd,ij = l1_multiply(X)
crossd,ij = l1_cross_distances(X)


cross = manhattan_distances(X,x)
multi = manhattan_multiply(X,x)
print multi.shape
print cross.shape

ns = w0+ np.sum( (v0.reshape(1, n_features))**2 * multi, axis=1) + sigmaf*np.exp(-0.5*np.sum( (l0.reshape(1, n_features))*2 * cross**2, axis=1))#15 combinaties van datapunten, de features worden gebundeld per waarde

#print ns
x = array2d(x)
print ns.shape

'''
gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1,
                     random_start=10)
y = np.array([1,4,9])
gp.fit(b, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
a,b,c,d = gp.predict(a, eval_MSE=True)
print a
'''