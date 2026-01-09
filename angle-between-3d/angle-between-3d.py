import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    # Your code here
    v = np.array(v)
    w = np.array(w)
    num = np.dot(v, w)
    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)

    if v_norm < 10e-10 or w_norm < 10e-10:
        return np.nan

    denom = v_norm * w_norm
    theta = np.arccos(np.clip(num / denom, -1, 1))
    return theta