import os
import sys

import numpy as np
import scipy

sys.path.append(os.environ["CD_CODE_DIR"])

from quspin.tools.misc import get_matvec_function

from ham_controls.build_controls import build_controls_mat


def build_Hcd(t, ham, AGPtype, ctrls, couplings, couplings_args):
    """Returns a list of (sparse or dense) matrices, which are within $H_{CD}$
    in order to do matrix-vector multiplication on the wavefunction
    Parameters:
        t (float):                  Time at which to build the Hamiltonian
        ham (Hamiltonian_CD):       Counterdiabatic Hamiltonian of interest
        AGPtype (str):              Either "commutator" or "krylov" depending
                                    on the type of AGP desired
        ctrls (list):               List of control Hamiltonians
        couplings (list):           List of coupling functions for control terms
        couplings_args (list):      List of list of arguments for the coupling functions

    """
    # need to get bare H, controls, and AGP term
    bareH = ham.bareH.tocsr(time=t) if ham.sparse else ham.bareH.toarray(time=t)
    dlamH = ham.dlamH.tocsr(time=t) if ham.sparse else ham.dlamH.toarray(time=t)
    Hmats = [bareH]
    for i in range(len(ctrls)):
        Hmats.append(build_controls_mat(ham, ctrls[i], couplings[i], couplings_args[i]))
    if ham.agp_order > 0:  # may want to evolve without AGP
        Hmats.append(ham.build_cd_term_mat(AGPtype, t, bareH, dlamH))
    return Hmats


def schro_RHS(t, psi, ham, AGPtype, ctrls, couplings, couplings_args):
    """Compute the right hand side of the Schrodinger equation,
    i.e. -i * H_cd * psi
    Parameters:
        t (float):                  Time at which to compute the RHS of SE
        ham (Hamiltonian_CD):       Counterdiabatic Hamiltonian of interest
        AGPtype (str):              Either "commutator" or "krylov" depending
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
    fname,
    AGPtype,
    ctrls,
    couplings,
    couplings_args,
    grid_size,
    init_state,
    save_states=False,
):
    """Computes the time evolution (according to the Schrodinger equation)
    of some initial state according to the given Hamiltonian
    Parameters:
        ham (Hamiltonian_CD):       Counterdiabatic Hamiltonian of interest
        fname (str):                Name of file to store instantaneous wavefunctions
        AGPtype (str):              Either "commutator" or "krylov" depending
                                    on the type of AGP desired
        ctrls (list):               List of control Hamiltonians
        couplings (list):           List of coupling functions for control terms
        couplings_args (list):      List of list of arguments for the coupling functions
        grid_size (int):            Number of time steps to take
        init_state (np.array):      Vector of initial wavefunction
        save_state (bool):          Whether to save the state at each time step
    """
    dt = ham.schedule.tau / grid_size
    tgrid = np.linspace(0, ham.schedule.tau, grid_size)
    sched = ham.schedule

    complex_ODE = scipy.integrate.ode(schro_RHS)
    complex_ODE.set_integrator("zvode")
    complex_ODE.set_initial_value(init_state, 0)
    # ctrls is e.g. ['yy', "HHV_VHV"], couplings is e.g. [sin_coupling, sin_coupling]
    # couplings_args is [[sched, ns, coeffs], [sched, ns, coeffs]]
    complex_ODE.set_f_params(ham, AGPtype, ctrls, couplings, couplings_args)  # TODO
    while complex_ODE.successful and complex_ODE.t < ham.schedule.tau:
        if save_states:
            path_name = "evolved_state_data/{0}_t{1:.6f}.txt"
            np.savetxt(path_name, complex_ODE.y)
        complex_ODE.integrate(complex_ODE.t + dt)
    return complex_ODE.y
