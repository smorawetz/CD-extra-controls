def build_agp_H_mat(t, ham, ctrls, couplings, couplings_args):
    """Build H for use in forming the AGP. This will consist of the
    bare Hamiltonian plus any extra controlsa, including HHV/VHV etc.
    Parameters:
        t (float):                          Time at which to build the AGP term
        ham (Hamiltonian_CD):               Counterdiabatic Hamiltonian of interest
        ctrls (listof str):                 List of control types to add
        couplings (listof function):        List of dlam_coupling functions for
                                            control terms
        couplings_args (listof list):       List of list of arguments for the coupling functions
    """
    H = ham.bareH.tocsr(time=t) if ham.sparse else ham.bareH.toarray(time=t)
    for i in range(len(ctrls)):
        H += build_controls_mat(ham, ctrls[i], couplings[i], couplings_args[i])
    return H


def build_agp_dlamH_mat(t, ham, ctrls, couplings, dlam_couplings, couplings_args):
    """Build dlamH for use in forming the AGP. This will consist of the
    dlam of the bare Hamiltonian plus any extra controls, including HHV/VHV etc.
    Parameters:
        t (float):                          Time at which to build the AGP term
        ham (Hamiltonian_CD):               Counterdiabatic Hamiltonian of interest
        ctrls (listof str):                 List of control types to add
        couplings (listof function):        List of dlam_coupling functions for
                                            control terms
        dlam_couplings (listof function):   List of dlam_coupling functions for
                                            control terms
        couplings_args (listof list):       List of list of arguments for the coupling functions
    """
    dlamH = ham.dlamH.tocsr(time=t) if ham.sparse else ham.dlamH.toarray(time=t)
    for i in range(len(ctrls)):
        if ctrls[i] == "HHV":  # have term from differentiating first H in [H, [H, V]]
            dlamH += build_controls_mat(ham, "VHV", couplings[i], couplings_args[i])
        dlamH += build_controls_mat(ham, ctrls[i], dlam_couplings[i], couplings_args[i])
    return dlamH
