import numpy as np
from sklearn.utils import array2d, check_random_state, check_arrays


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

def squared_exponential(theta, d,multiplyd):
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
        
        
def non_stationary(theta,crossd,multiplyd):
    theta = np.asarray(theta, dtype=np.float)
    crossd = np.asarray(crossd, dtype=np.float)
    multiplyd = np.asarray(multiplyd, dtype=np.float)

    if crossd.shape != multiplyd.shape:
        raise ValueError("cross en multiply moeten zelfde dimensies hebben")
    if crossd.ndim > 1:
        n_features = crossd.shape[1]
    else:
        n_features = 1


    if theta.size != (2*n_features+2):
        raise ValueError("Length of theta must be 1 or %s" % (2*n_features+2))
    else:
        w0=theta[0]
        v0=theta[1:(n_features+1)]
        l0=theta[(n_features+1):(2*n_features+1)]
        sigmaf=theta[(2*n_features+1)]
        
        
        return w0+ np.sum( v0.reshape(1, n_features)* multiplyd, axis=1) + sigmaf*np.exp(-0.5*np.sum( l0.reshape(1, n_features)* crossd** 2, axis=1))
        
theta0 = np.array([1e-1,1e-1,1e-1,1e-1,1e-1,1e-1,1e-1,1e-1])
#theta0 = np.array([1e-1])
b = [[1,2,4],[4,5,6],[7,8,9],[10,11,12]] #trainingsset: 4 datapunten, 3 features
b = np.array(b)
print b

cross,ij = l1_cross_distances(b)
mult = l1_multiply(b)[0]
#print squared_exponential(theta0,cross,mult)
print non_stationary(theta0,cross,mult)  

r = non_stationary(theta0,cross,mult)  
R = np.eye(4)
R[ij[:, 0], ij[:, 1]] = r
R[ij[:, 1], ij[:, 0]] = r
print(R)