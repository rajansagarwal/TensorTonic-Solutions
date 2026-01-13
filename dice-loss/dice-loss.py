import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Write code here
    p, y = np.array(p).flatten(), np.array(y).flatten()
    n = 2 * np.sum(p * y) + eps
    d = np.sum(p) + np.sum(y) + eps

    return 1 - (n/d)