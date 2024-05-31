import os
import sys

import numpy as np
import scipy  # TODO TEMP

sys.path.append(os.environ["CD_CODE_DIR"])

from tools.lin_alg_calls import calc_comm

DIV_EPS = 1e-16


def op_norm(op, basis_size, norm_type, gstate=None):
    """Calculate the norm of an operator. Will be either the trace
    (infinite temperature) or the ground state expectation value
    (zero temperature) of the input operator
    Parameters:
        op (sparse or dense array):         Operator to calculate the norm of
        basis_size (int):                   Size of the Hilbert space
        gstate (np.array):                  Ground state of the Hamiltonian to use in
                                            zero temperature optimization
    """
    if norm_type == "trace":
        return np.sqrt((op.conj().T @ op).trace() / basis_size).item().real
    elif norm_type == "ground_state":
        return np.sqrt(gstate.conj().dot((op.conj().T @ op).dot(gstate))).item().real
    else:
        raise ValueError("Invalid norm type")


def get_lanc_coeffs(agp_order, Hmat, dlamHmat, basis_size, norm_type, gstate=None):
    """Calculate the Lanczos coefficients for the for the action of the
    Liouvillian L = [H, .] on dlamH at a given time
    Parameters:
        agp_order (int):                    Order of the AGP ansatz
        Hmat (sparse or dense array):       Hamiltonian matrix at some time
        dlamHmat (sparse or dense array):   Initial operator to start the Lanczos iteration
        basis_size (int):                   Size of the basis in the Hilbert space
        norm_type (str):                    Either "trace" or "ground_state" for the norm
        gstate (np.array):                  Ground state of the Hamiltonian to use in
                                            zero temperature optimization
    """
    lanc_coeffs = []
    O0 = dlamHmat.copy()
    b0 = op_norm(O0, basis_size, norm_type, gstate)
    O0 /= b0 + DIV_EPS
    lanc_coeffs.append(b0)
    O1 = calc_comm(Hmat, O0)
    b1 = op_norm(O1, basis_size, norm_type, gstate)
    O1 /= b1 + DIV_EPS
    lanc_coeffs.append(b1)
    for n in range(2, 2 * agp_order + 1):
        On = calc_comm(Hmat, O1) - lanc_coeffs[n - 1] * O0
        bn = op_norm(On, basis_size, norm_type, gstate)
        On /= bn + DIV_EPS
        lanc_coeffs.append(bn)
        O0 = O1
        O1 = On
    return np.array(lanc_coeffs)


def calc_A_val(lanc_coeffs, k):
    """Compute the coefficients defined by $A_k = b_{2k}^2 + b_{2k-1}^2$ at some time
    Parameters:
        lanc_coeffs (np.array):     Array of Lanczos coefficients at some time
        k (int):                    Which $A_k$ to calculate
    """
    coeffs = lanc_coeffs[(2 * k - 1) : (2 * k + 1)]  #  has b_{2k-1} and b_{2k}
    return np.sum(coeffs**2)


def calc_B_val(lanc_coeffs, k):
    """Compute the coefficients defined by $B_k = b_{2k}b_{2k+1}$ at some time
    Parameters:
        lanc_coeffs (np.array):     Array of Lanczos coefficients at some time
        k (int):                    Which $B_k$ to calculate
    """
    coeffs = lanc_coeffs[(2 * k) : (2 * k + 2)]  #  has b_{2k} and b_{2k+1}
    return np.prod(coeffs)


def calc_r_vals(lanc_coeffs, Aks, Bks, agp_order):
    """Compute the coefficients $r_k$ which give $\gamma_{k+1} = -r_k \gamma_k$,
    and can be solved recursively, beginning with $r_{N_1} = B_{N-1}/A_N$ and
    defined by recursion relation $r_k = B_{k-1}/(A_k - B_k r_k)$
    Parameters:
        lanc_coeffs (np.array):     Array of Lanczos coefficients at some time
        Aks (np.array):             Coefficients $A_k$ required to compute $r_k$
        Bks (np.array):             Coefficients $B_k$ required to compute $r_k$
        agp_order (int):            Order of AGP expansion
    """
    if agp_order == 1:  # no recursion, and r_1 = 0 needed in gamma_1 calculation
        return [0]
    r_vals = [Bks[-1] / Aks[-1]]
    for k in range(agp_order - 2, 0, -1):  # step backwards
        r_vals.insert(0, Bks[k - 1] / (Aks[k] - Bks[k] * r_vals[0]))
    return r_vals


def get_gamma_vals(lanc_coeffs, agp_order):
    """Compute the coefficients of the AGP in the Krylov space construction,
    denoted $\gamma_k$ in the paper. The coefficients are defined by the
    $\gamma_1 = -b_0 b_1 / (A_1 - r_1 B_1)$ and $\gamma_{k+1} = - r_k \gamma_k$
    Parameters:
        lanc_coeffs (np.array):     Array of Lanczos coefficients at some time
        agp_order (int):            Order of AGP expansion
    """
    Aks = []
    Bks = []
    for k in range(1, agp_order):
        Aks.append(calc_A_val(lanc_coeffs, k))
        Bks.append(calc_B_val(lanc_coeffs, k))
    Aks.append(calc_A_val(lanc_coeffs, agp_order))  # also need A_{agp_order}
    r_vals = calc_r_vals(lanc_coeffs, Aks, Bks, agp_order)
    b0b1 = np.prod(lanc_coeffs[0:2])
    if agp_order == 1:
        gammas = [-b0b1 / Aks[0]]
    else:
        gammas = [-b0b1 / (Aks[0] - r_vals[0] * Bks[0])]
    for k in range(1, agp_order):
        gammas.append(-r_vals[k - 1] * gammas[-1])
    return np.array(gammas)
