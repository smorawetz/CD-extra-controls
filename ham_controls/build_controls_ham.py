import os
import sys

import pickle

import numpy as np
import scipy

sys.path.append(os.environ["CD_CODE_DIR"])

from ham_controls.build_controls import build_controls_mat

with open("{0}/dicts/couplings.pkl".format(os.environ["CD_CODE_DIR"]), "rb") as f:
    couplings_dict = pickle.load(f)

with open("{0}/dicts/dlam_couplings.pkl".format(os.environ["CD_CODE_DIR"]), "rb") as f:
    dlam_couplings_dict = pickle.load(f)


def build_H_controls_mat(ham, t, ctrls, couplings, couplings_args):
    """Build H for use in forming the AGP. This will consist of the
    bare Hamiltonian plus any extra controlsa, including HHV/VHV etc.
    Parameters:
        ham (Hamiltonian_CD):               Counterdiabatic Hamiltonian of interest
        t (float):                          Time at which to build the AGP term
        ctrls (listof str):                 List of control types to add
        couplings (listof str):             List of strings corresponding to coupling
                                            functions of control terms
        couplings_args (listof list):       List of list of arguments for the coupling functions
    """
    H = ham.bareH.tocsr(time=t) if ham.sparse else ham.bareH.toarray(time=t)
    for i in range(len(ctrls)):
        H += build_controls_mat(
            t,
            ham,
            ctrls[i],
            couplings_dict[couplings[i]],
            couplings_args[i],
        )
    return H


def build_dlamH_controls_mat(ham, t, ctrls, couplings, couplings_args):
    """Build dlamH for use in forming the AGP. This will consist of the
    dlam of the bare Hamiltonian plus any extra controls, including HHV/VHV etc.
    Parameters:
        ham (Hamiltonian_CD):               Counterdiabatic Hamiltonian of interest
        t (float):                          Time at which to build the AGP term
        ctrls (listof str):                 List of control types to add
        couplings (listof str):             List of strings corresponding to coupling
                                            functions of control terms
        couplings_args (listof list):       List of list of arguments for the coupling functions
    """
    dlamH = ham.dlamH.tocsr(time=t) if ham.sparse else ham.dlamH.toarray(time=t)
    for i in range(len(ctrls)):
        if ctrls[i] == "HHV":  # have term from differentiating first H in [H, [H, V]]
            dlamH += build_controls_mat(
                t,
                ham,
                "VHV",
                couplings_dict[couplings[i]],
                couplings_args[i],
            )
        dlamH += build_controls_mat(
            t,
            ham,
            ctrls[i],
            dlam_couplings_dict[couplings[i]],
            couplings_args[i],
        )
    return dlamH


def get_H_controls_gs(ham, t, ctrls, couplings, couplings_args):
    """Get the ground state of the Hamiltonian matrix, including the
    extra controls, at time t
    Parameters:
        ham (Hamiltonian_CD):               Counterdiabatic Hamiltonian of interest
        t (float):                          Time at which to build the AGP term
        ctrls (listof str):                 List of control types to add
        couplings (listof str):             List of strings corresponding to coupling
                                            functions of control terms
        couplings_args (listof list):       List of list of arguments for the coupling functions
    """
    H = build_H_controls_mat(ham, t, ctrls, couplings, couplings_args)
    if ham.sparse:
        eigsolver = scipy.sparse.linalg.eigsh
        kwargs = {"k": 1, "which": "SA"}
    else:
        eigsolver = np.linalg.eigh
        kwargs = {}
    eigvals, eigvecs = eigsolver(H, **kwargs)
    idx = eigvals.argsort()[0]
    inst_gs = eigvecs[:, idx]
    return inst_gs.reshape(-1)
