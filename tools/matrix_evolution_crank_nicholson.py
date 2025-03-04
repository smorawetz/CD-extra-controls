import os
import sys

import numpy as np
import scipy

sys.path.append(os.environ["CD_CODE_DIR"])

from quspin.tools.misc import get_matvec_function

from ham_controls.build_controls_ham import (
    build_H_controls_mat,
    build_dlamH_controls_mat,
)
from ham_controls.build_controls import build_controls_mat


def build_Hcd(t, ham, AGPtype, ctrls, ctrls_couplings, ctrls_args):
    """Returns a list of (sparse or dense) matrices, which are within $H_{CD}$
    in order to do matrix-vector multiplication on the wavefunction
    Parameters:
        t (float):                      Time at which to build the Hamiltonian
        ham (Hamiltonian_CD):           Counterdiabatic Hamiltonian of interest
        AGPtype (str):                  Type of approximate AGP to construct depending
                                        on the type of AGP desired
        ctrls (listof str):             List of control Hamiltonians
        ctrls_couplings (listof str):   List of strings indexing coupling functions
                                        for control terms
        ctrls_args (list):              List of list of arguments for the coupling
                                        functions

    """
    # need to get bare H, controls, and AGP term
    bareH = build_H_controls_mat(ham, t, ctrls, ctrls_couplings, ctrls_args)
    dlamH = build_dlamH_controls_mat(ham, t, ctrls, ctrls_couplings, ctrls_args)
    Hmats = [bareH]
    if ham.agp_order > 0:  # may want to evolve without AGP
        Hmats.append(ham.build_cd_term_mat(t, AGPtype, bareH, dlamH))
    return Hmats


def CN_step(t, dt, psi, ham, AGPtype, ctrls, couplings, couplings_args):
    """Compute one step in Crank-Nicolson evolution
    i.e. -i * H_cd * psi
    Parameters:
        t (float):                  Time at which to compute the RHS of SE
        dt (float):                 Time step size
        ham (Hamiltonian_CD):       Counterdiabatic Hamiltonian of interest
        AGPtype (str):              Type of approximate AGP to construct depending
                                    on the type of AGP desired
        ctrls (list):               List of control Hamiltonians
        couplings (list):           List of coupling functions for control terms
        couplings_args (list):      List of list of arguments for the coupling functions
        psi (np.array):             Wavefunction to evolve
    """
    H = sum(build_Hcd(t, ham, AGPtype, ctrls, couplings, couplings_args))
    I = np.eye(H.shape[0])
    psi = psi.reshape(len(psi), 1)
    A = I - 1j * dt / 2 * H
    B = I + 1j * dt / 2 * H
    return np.linalg.solve(A, (B @ psi))


def do_evolution(
    ham,
    AGPtype,
    ctrls,
    couplings,
    couplings_args,
    grid_size,
    init_state,
    dt=0.01,
):
    """Computes the time evolution (according to the Schrodinger equation)
    of some initial state according to the given Hamiltonian
    Parameters:
        ham (Hamiltonian_CD):       Counterdiabatic Hamiltonian of interest
        AGPtype (str):              Type of approximate AGP to construct depending
                                    on the type of AGP desired
        ctrls (list):               List of control Hamiltonians
        couplings (list):           List of coupling functions for control terms
        couplings_args (list):      List of list of arguments for the coupling functions
        grid_size (int):            Number of time steps to take
        init_state (np.array):      Vector of initial wavefunction
        dt (float):                 Time step size for Crank-Nicolson
    """
    tgrid = np.linspace(0, ham.schedule.tau, grid_size)
    sched = ham.schedule

    psi = init_state
    t = 0
    wfs_full, ts_full = [], []
    i = 0
    while t < sched.tau:
        if i % (ham.schedule.tau / dt / grid_size) == 0:
            wfs_full.append(np.asarray(psi).flatten())
            ts_full.append(t)
        psi = CN_step(t, dt, psi, ham, AGPtype, ctrls, couplings, couplings_args)
        t += dt
        i += 1
    return np.array(ts_full), np.array(wfs_full)
