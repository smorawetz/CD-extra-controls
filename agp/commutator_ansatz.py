import os
import sys

import numpy as np

sys.path.append(os.environ["CD_CODE_DIR"])

from tools.lin_alg_calls import calc_comm


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
    for k in range(2 * agp_order):
        commutators.append(calc_comm(H, commutators[-1]))
    return commutators


def inf_temp_R(k, commutators):
    """Compute the spectral function moments (denoted in notes as $R_k$)
    matrix for the infinite temperature AGP. Works for both sparse and
    dense matrices
    Parameters:
        k (int):                    Compute k-th moment of the spectral function
        commutators (list):         List of commutators of H and dlam_H
    """
    return (commutators[k].conj().T @ commutators[k]).trace().item().real


def zero_temp_R(k, commutators, gstate):
    """Compute the R matrix for the zero temperature (ground state)
    AGP. Works for both sparse and dense matrices
    Parameters:
        k (int):                    Order of AGP commuator ansatz
        commutators (list):         List of commutators of H and dlam_H
    """
    return (
        gstate.dot((commutators[k].conj().T @ commutators[k])).dot(gstate).item().real
    )


def compute_Rs(agp_order, calc_R, commutators, args):
    """Compute the R matrix and R vector used in computing the
    alpha coefficients of the AGP ansatz
    Parameters:
        agp_order (int):            Order of AGP commuator ansatz
        calc_R (function):          Function to compute R matrix
        commutators (list):         List of commutators of H and dlam_H
        args (tuple):               Arguments to pass to calc_R (i.e. `gstate`)

    """
    R_list = []
    for k in range(2 * agp_order + 1):
        R_list.append(calc_R(k, commutators, *args))
    Rmat = np.zeros((agp_order, agp_order))
    Rvec = np.zeros(agp_order)
    for i in range(agp_order):
        Rvec[i] = R_list[i + 1]
        for j in range(agp_order):
            Rmat[i, j] = R_list[i + j + 2]
    return Rmat, Rvec


def combine_Rs_blocks(listof_Rmats, listof_Rvecs):
    """Combine a list of R matrices and R vectors corresponding to
    those in separate blcoks (of Hamiltonian) into a single R matrix and
    R vector for use in the alpha coefficient calculation
    Parameters:
        listof_Rmats (list):        List of R matrices
        listof_Rvecs (list):        List of R vectors
    """
    Rmat, Rvec = listof_Rmats[0], listof_Rvecs[0]
    for j in range(1, len(listof_Rvecs)):
        Rmat = Rmat + listof_Rmats[j]
        Rvec = Rvec + listof_Rvecs[j]
    return Rmat, Rvec


def get_alphas(agp_order, H, dlam_H, norm_type, gstate=None):
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
        args = []
    elif norm_type == "ground_state":
        calc_R = zero_temp_R
        args = [gstate]
    Rmat, Rvec = compute_Rs(agp_order, calc_R, commutators, args)
    return np.linalg.solve(Rmat, -Rvec)
