def reduced_likelihood_function(self, theta=None):
        """
        This function determines the BLUP parameters and evaluates the reduced
        likelihood function for the given autocorrelation parameters theta.

        Maximizing this function wrt the autocorrelation parameters theta is
        equivalent to maximizing the likelihood of the assumed joint Gaussian
        distribution of the observations y evaluated onto the design of
        experiments X.

        Parameters
        ----------
        theta : array_like, optional
            An array containing the autocorrelation parameters at which the
            Gaussian Process model parameters should be determined.
            Default uses the built-in autocorrelation parameters
            (ie ``theta = self.theta_``).

        Returns
        -------
        reduced_likelihood_function_value : double
            The value of the reduced likelihood function associated to the
            given autocorrelation parameters theta.

        par : dict
            A dictionary containing the requested Gaussian Process model
            parameters:

                sigma2
                        Gaussian Process variance.
                beta
                        Generalized least-squares regression weights for
                        Universal Kriging or given beta0 for Ordinary
                        Kriging.
                gamma
                        Gaussian Process weights.
                C
                        Cholesky decomposition of the correlation matrix [R].
                Ft
                        Solution of the linear equation system : [R] x Ft = F
                G
                        QR decomposition of the matrix Ft.
        """

        if theta is None:
            # Use built-in autocorrelation parameters
            theta = self.theta0
        # Initialize output
        reduced_likelihood_function_value = - np.inf
        par = {}

        # Retrieve data
        n_samples = self.X.shape[0]
        D = self.D
        ij = self.ij
        F = self.F

        if D is None:
            # Light storage mode (need to recompute D, ij and F)
            D, ij = l1_cross_distances(self.X)
            if (np.min(np.sum(D, axis=1)) == 0.
                    and self.corr != correlation.pure_nugget):
                raise Exception("Multiple X are not allowed")
            F = self.regr(self.X)

        # Set up R
        r = self.corr(theta, D)
        R = np.eye(n_samples) * (1. + self.nugget)
        R[ij[:, 0], ij[:, 1]] = r
        R[ij[:, 1], ij[:, 0]] = r

        # Cholesky decomposition of R
        try:
            C = linalg.cholesky(R, lower=True)
        except linalg.LinAlgError:
            return reduced_likelihood_function_value, par

        # Get generalized least squares solution
        Ft = solve_triangular(C, F, lower=True)
        try:
            Q, G = linalg.qr(Ft, econ=True)
        except:
            #/usr/lib/python2.6/dist-packages/scipy/linalg/decomp.py:1177:
            # DeprecationWarning: qr econ argument will be removed after scipy
            # 0.7. The economy transform will then be available through the
            # mode='economic' argument.
            Q, G = linalg.qr(Ft, mode='economic')
            pass

        sv = linalg.svd(G, compute_uv=False)
        rcondG = sv[-1] / sv[0]
        if rcondG < 1e-10:
            # Check F
            sv = linalg.svd(F, compute_uv=False)
            condF = sv[0] / sv[-1]
            if condF > 1e15:
                raise Exception("F is too ill conditioned. Poor combination "
                                "of regression model and observations.")
            else:
                # Ft is too ill conditioned, get out (try different theta)
                return reduced_likelihood_function_value, par

        Yt = solve_triangular(C, self.y, lower=True)
        if self.beta0 is None:
            # Universal Kriging
            beta = solve_triangular(G, np.dot(Q.T, Yt))
        else:
            # Ordinary Kriging
            beta = np.array(self.beta0)

        rho = Yt - np.dot(Ft, beta)
        sigma2 = (rho ** 2.).sum(axis=0) / n_samples
        # The determinant of R is equal to the squared product of the diagonal
        # elements of its Cholesky decomposition C
        detR = (np.diag(C) ** (2. / n_samples)).prod()

        # Compute/Organize output
        reduced_likelihood_function_value = - sigma2.sum() * detR
        par['sigma2'] = sigma2 * self.y_std ** 2.
        par['beta'] = beta
        par['gamma'] = solve_triangular(C.T, rho)
        par['C'] = C
        par['Ft'] = Ft
        par['G'] = G
        
        print(reduced_likelihood_function_value)
        return reduced_likelihood_function_value, par