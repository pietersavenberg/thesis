import correlation_models
import numpy as np
from sklearn.utils import array2d

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
        D[ll_0:ll_1] = (X[k] * X[(k + 1):n_samples])

    return D, ij.astype(np.int)
    
    
#X = [-1.6803361, -0.84016805, 0. , 0.42008403, 0.84016805, 1.26025208]
X = np.array([[-1.6803361],[ -0.84016805],[ 0.] ,[ 0.42008403],[ 0.84016805], [1.26025208]])
params = 10. ** np.asarray([-0.30103, 0.69897, -0.30103,  0.69897])
crossd=np.array([[ 0.84016805],
       [ 1.6803361 ],
       [ 2.10042013],
       [ 2.52050415],
       [ 2.94058818],
       [ 0.84016805],
       [ 1.26025208],
       [ 1.6803361 ],
       [ 2.10042013],
       [ 0.42008403],
       [ 0.84016805],
       [ 1.26025208],
       [ 0.42008403],
       [ 0.84016805],
       [ 0.42008403]])

multiplyd,ij = l1_multiply(X)

def nonstationary(xi, xj):
    l = params[2]
    sigma_f = params[3]
    v = params[1]
    w0 = params[0]
    print w0, v, sigma_f, l
    return w0 + np.sum(xi*xj) / (v ** 2) + sigma_f * np.exp(-0.5*np.sum((xi-xj) ** 2) / l ** 2)

print nonstationary(5, 5)

#K = np.asarray([[nonstationary(xi, xj) for xi in X] for xj in X])
#print K
#
#
#l0 = params[2]
#sigmaf = params[3]
#v0 = params[1]
#w0 = params[0]
#n_features = 1
#n_samples = 6
#
#r= w0+ np.sum( 1/(v0.reshape(1, n_features))**2 * multiplyd, axis=1) + sigmaf*np.exp(-0.5*np.sum( 1/(l0.reshape(1, n_features))**2 * crossd**2, axis=1))
#
#
#R = np.eye(n_samples) 
#R[ij[:, 0], ij[:, 1]] = r
#R[ij[:, 1], ij[:, 0]] = r
#        #if self.corr == 'non_stationary':
#         #   print("deze shit moet uitgevoerd worden")
#for i in range(n_samples):
#    som = 0.
#    for j in range(n_features):
#        som += X[i,j]**2 / (v0**2)
#    R[i,i] += w0 + som
#        
#print R
