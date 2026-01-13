import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    diff = np.abs(y_true - y_pred)
    return np.mean(np.where(diff <= delta, 1/2 * np.square(diff), delta * (diff - 1/2 * delta)))
