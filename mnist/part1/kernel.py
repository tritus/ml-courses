import numpy as np

### Functions for you to fill in ###

# pragma: coderesponse template


def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    return np.power(np.matmul(X, Y.transpose()) + c, p)
# pragma: coderesponse end

# pragma: coderesponse template


def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    (n, d) = X.shape
    m = Y.shape[0]
    gy = np.repeat(Y[np.newaxis, :], X.shape[0], axis=0)
    xprime = np.reshape(X[np.newaxis, :],(X.shape[0],1,X.shape[1]))
    gx = np.repeat(xprime, Y.shape[0], axis=1)
    diff = gy-gx
    reshaped_diff1 = np.reshape(diff[np.newaxis, :], (n, m, 1, d))
    reshaped_diff2 = np.reshape(diff[np.newaxis, :], (n, m, d, 1))
    sq_norms = np.matmul(reshaped_diff1, reshaped_diff2)
    nm_sq_norms = np.squeeze(sq_norms, axis=(2,3))  
    return np.exp(-gamma*nm_sq_norms)
# pragma: coderesponse end
