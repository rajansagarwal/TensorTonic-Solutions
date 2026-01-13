import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pos = np.expand_dims(np.arange(seq_len), axis=1)

    result = np.arange(d_model)

    div = np.power(base, 2 * (result // 2) / d_model)
    angles = pos / div

    return np.where(result % 2 == 0, np.sin(angles), np.cos(angles))


def add_positional_encoding(x, base=10000.0):
    """
    Add PE to input x of shape (B, T, d_model); return same shape.
    """
    # Write code here
    b, t, d = x.shape
    pe = positional_encoding(t, d, base)
    return x + pe