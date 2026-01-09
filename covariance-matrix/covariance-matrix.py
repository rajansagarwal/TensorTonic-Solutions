import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X_n = np.array(X)
    
    if X_n.ndim != 2:
        return None
    
    N, _ = X_n.shape

    if N < 2:
        return None

    mn = np.mean(X_n, axis=0)
    x_cen = X_n - mn

    result = (1/(N-1)) * x_cen.T @ x_cen
    return result