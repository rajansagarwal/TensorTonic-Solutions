import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    # Write code here
    p = np.array(p)
    q = np.array(q)
    q_stable = q + eps 
    return np.sum(p[p>0] * np.log(p[p>0] / q_stable[p>0]), 0)