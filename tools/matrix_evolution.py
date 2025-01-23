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


def build_Hcd_FE(t, ham, omega, ctrls, ctrls_couplings, ctrls_args):
    """Returns a list of (sparse or dense) matrices, which are within $H_{CD}$
    in order to do matrix-vector multiplication on the wavefunction
    Parameters:
        t (float):                      Time at which to build the Hamiltonian
        ham (Hamiltonian_CD):           Counterdiabatic Hamiltonian of interest
        omega (float):                  The Floquet frequency in the Floquet
                                        -engineered Hamiltonian
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
        Hmats.append(ham.build_FE_cd_term_mat(t, omega, bareH, dlamH))
    return Hmats


def schro_RHS(t, stack_psi, ham, AGPtype, ctrls, couplings, couplings_args, omega=None):
    """Compute the right hand side of the Schrodinger equation,
    i.e. -i * H_cd * psi
    Parameters:
        t (float):                  Time at which to compute the RHS of SE
        stack_psi (np.array):       Wavefunction to evolve, with stacked real
                                    (first N elts) and imaginary (last N elts) parts
        ham (Hamiltonian_CD):       Counterdiabatic Hamiltonian of interest
        AGPtype (str):              Type of approximate AGP to construct depending
                                    on the type of AGP desired
        ctrls (list):               List of control Hamiltonians
        couplings (list):           List of coupling functions for control terms
        couplings_args (list):      List of list of arguments for the coupling functions
        omega (float):              The Floquet frequency if realizing by Floquet engineering
    """
    if AGPtype == "floquet":
        Hcd = build_Hcd_FE(t, ham, omega, ctrls, couplings, couplings_args)
    else:
        Hcd = build_Hcd(t, ham, AGPtype, ctrls, couplings, couplings_args)
    N = len(stack_psi) // 2
    stack_psi = stack_psi.reshape(len(stack_psi), 1)
    delta_psi = np.zeros_like(stack_psi)
    for H in Hcd:
        real_psi_mult = H @ stack_psi[:N]
        imag_psi_mult = H @ stack_psi[N:]
        delta_psi[:N] += real_psi_mult.imag + imag_psi_mult.real
        delta_psi[N:] += imag_psi_mult.imag - real_psi_mult.real
    # print(t, delta_psi[0] + 1j * delta_psi[N], delta_psi[1] + 1j * delta_psi[N + 1])
    return delta_psi


def do_evolution(
    ham, AGPtype, ctrls, couplings, couplings_args, grid_size, init_state, omega=None
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
        omega (float):              The Floquet frequency if realizing by Floquet engineering
    """
    dt = ham.schedule.tau / grid_size
    tgrid = np.linspace(0, ham.schedule.tau, grid_size)
    sched = ham.schedule

    N = len(init_state)
    stack_psi = np.hstack((init_state.real, init_state.imag))

    real_ODE = scipy.integrate.ode(schro_RHS)
    # real_ODE.set_integrator("vode", method="bdf")
    real_ODE.set_integrator("vode")
    real_ODE.set_initial_value(stack_psi, 0)
    real_ODE.set_f_params(ham, AGPtype, ctrls, couplings, couplings_args, omega)
    wfs_full, ts_full = [], []
    while real_ODE.successful and real_ODE.t < ham.schedule.tau:
        wfs_full.append(real_ODE.y[:N] + 1j * real_ODE.y[N:])
        ts_full.append(real_ODE.t)
        real_ODE.integrate(real_ODE.t + dt)
    return np.array(ts_full), np.array(wfs_full)
