import numpy as np


def calc_comm(A, B):
    """Returns the commutator of two matrices $[A, B] = AB - BA
    Parameters:
        A (np.array):   Matrix A
        B (np.array):   Matrix B
    """
    return A @ B - B @ A


def calc_fid(psi1, psi2):
    """Calculates the fidelity between two states psi1 and psi2
    Parameters:
        psi1 (np.array):    First state
        psi2 (np.array):    Second state
    """
    norm1 = np.abs(np.dot(np.conj(psi1), psi1))
    norm2 = np.abs(np.dot(np.conj(psi2), psi2))
    return np.abs(np.dot(np.conj(psi1), psi2)) ** 2 / norm1 / norm2
