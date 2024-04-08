import numpy as np


def compute_commutators(agp_order, H, dlam_H):
    """Compute the nested commutators of H and dlam_H up to
    the requisite order of the AGP, works for sparse or dense
    H and dlam_H
    Parameters:
        agp_order (int):                    Order of AGP commuator ansatz
        H (dense or sparse array):          Hamiltonian matrix
        dlam_H (dense or sparse array):     dlam_H matrix
    """
    commutators = [dlam_H]
    for k in range(agp_order):
        commutators.append(H @ commutators[-1] - commutators[-1] @ H)
    return commutators


def inf_temp_R(k, commutators):
    """Compute the R matrix for the infinite temperature AGP. Works
    for both sparse and dense matrices
    Parameters:
        k (int):                    Order of AGP commuator ansatz
        commutators (list):         List of commutators of H and dlam_H
    """
    return (-1) ** k * (commutators[k].conj().T @ commutators[k]).trace()


def zero_temp_R(k, commutators, gstate):
    """Compute the R matrix for the zero temperature (ground state)
    AGP. Works for both sparse and dense matrices
    Parameters:
        k (int):                    Order of AGP commuator ansatz
        commutators (list):         List of commutators of H and dlam_H
    """
    return (-1) ** k * gstate.dot(
        (commutators[k].conj().T @ commutators[k]).dot(gstate)
    )


def compute_Rs(agp_order, calc_R, args):
    """Compute the R matrix and R vector used in computing the
    alpha coefficients of the AGP ansatz
    Parameters:
        agp_order (int):                Order of AGP commuator ansatz
        calc_R (function):          Function to compute R matrix
        args (tuple):               Arguments to pass to calc_R
    """
    R_list = []
    for k in range(2 * agp_order + 1):
        R_list.append(calc_R(k, *args))
    Rmat = np.zeros(agp_order, agp_order)
    Rvec = np.zeros(agp_order)
    for i in range(agp_order):
        Rvec[i] = R_list[i + 1]
        for j in range(agp_order):
            Rmat[i, j] = R_list[i + j + 2]
    return Rmat, Rvec


def compute_alphas(agp_order, H, dlam_H, norm_type, gstate=None):
    """Compute the alphas for the AGP ansatz given a Hamiltonian matrix
    and a matrix for dlam_H. Can compute for both infinite temperature
    and zero temperature norm as required
    Parameters:
        agp_order (int):                Order of AGP commuator ansatz
        H (dense or sparse array):      Hamiltonian matrix
        dlam_H (dense or sparse array): dlam_H matrix
        norm_type (str):                Type of norm to compute alphas for, can
                                        be either "trace" or "ground_state"
        gstate (array):                 Ground state of the system, only required
                                        for zero_tempearture norm
    """
    commutators = compute_commutators(agp_order, H, dlam_H)
    if norm_type == "trace":
        calc_R = inf_temp_R
        args = (commutators,)
    elif norm_type == "ground_state":
        calc_R = zero_temp_R
        args = (commutators, gstate)
    Rmat, Rvec = compute_Rs(agp_order, calc_R, args)
    return np.linalg.solve(Rmat, -Rvec)
