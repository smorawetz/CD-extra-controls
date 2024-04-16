import numpy as np


def calc_comm(A, B):
    """Returns the commutator of two matrices $[A, B] = AB - BA
    Parameters:
        A (np.array):   Matrix A
        B (np.array):   Matrix B
    """
    return A @ B - B @ A
