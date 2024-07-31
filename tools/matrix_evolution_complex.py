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


def schro_RHS(t, psi, ham, AGPtype, ctrls, couplings, couplings_args):
    """Compute the right hand side of the Schrodinger equation,
    i.e. -i * H_cd * psi
    Parameters:
        t (float):                  Time at which to compute the RHS of SE
        ham (Hamiltonian_CD):       Counterdiabatic Hamiltonian of interest
        AGPtype (str):              Type of approximate AGP to construct depending
                                    on the type of AGP desired
        ctrls (list):               List of control Hamiltonians
        couplings (list):           List of coupling functions for control terms
        couplings_args (list):      List of list of arguments for the coupling functions
        psi (np.array):             Wavefunction to evolve
    """
    Hcd = build_Hcd(t, ham, AGPtype, ctrls, couplings, couplings_args)
    psi = psi.reshape(len(psi), 1)
    delta_psi = np.zeros_like(psi)
    for H in Hcd:
        matvec = get_matvec_function(H)
        delta_psi += -1j * matvec(H, psi)
    return delta_psi


def do_evolution(
    ham,
    AGPtype,
    ctrls,
    couplings,
    couplings_args,
    grid_size,
    init_state,
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
    """
    dt = ham.schedule.tau / grid_size
    tgrid = np.linspace(0, ham.schedule.tau, grid_size)
    sched = ham.schedule

    complex_ODE = scipy.integrate.ode(schro_RHS)
    complex_ODE.set_integrator("zvode")
    complex_ODE.set_initial_value(init_state, 0)
    complex_ODE.set_f_params(ham, AGPtype, ctrls, couplings, couplings_args)
    wfs_full, ts_full = [], []
    while complex_ODE.successful and complex_ODE.t < ham.schedule.tau:
        wfs_full.append(complex_ODE.y)
        ts_full.append(complex_ODE.t)
        complex_ODE.integrate(complex_ODE.t + dt)
    return np.array(ts_full), np.array(wfs_full)
