import numpy as np
from scipy.special import factorial

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    pmf = (np.exp(-lam) * (lam) ** k) / factorial(k)
    
    e_fact = lambda x, i: x ** i
    cum = np.arange(0, k + 1)
    cdf = np.sum(np.exp(-lam) * e_fact(lam, cum) / factorial(cum))
    return pmf, cdf

