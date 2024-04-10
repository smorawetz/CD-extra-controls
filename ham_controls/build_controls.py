from spin_half_controls import spin_half_controls


def build_controls_direct(ham, ctrl, coupling, coupling_args):
    """Build the extra controls for the Hamiltonian, which are the
    counterdiabatic terms
    Parameters:
        ham (Hamiltonian_CD):   Counterdiabatic Hamiltonian of interest
        ctrl (str):             Type of control to add
        coupling (function):    Coupling function for the control term
        coupling_args (list):   Arguments for the coupling function
    """
    if ham.model_type == "spin-1/2":
        control_term = spin_half_controls(len(ctrl), ham.model_dim)
    else:
        raise ValueError("Need code for {0} direct controls".format(ham.model_type))
    return control_term(ham, ctrl, coupling, coupling_args)  # a quspin hamiltonian


def build_controls_direct_mat(ham, ctrl, coupling, coupling_args):
    """Build the extra controls for the Hamiltonian, which are the
    counterdiabatic terms
    Parameters:
        ham (Hamiltonian_CD):       Counterdiabatic Hamiltonian of interest
        ctrl (str):                 Type of control to add
        coupling (function):    Coupling function for the control term
        coupling_args (list):   Arguments for the coupling function
    """
    # needs sinusoidal controls, and also commutators. add control terms file?
    ctrl_ham = build_controls_direct(ham, ctrl, coupling, coupling_args)
    return ctrl_ham.tocsr(time=t) if ham.sparse else ctrl_ham.toarray(time=t)


def build_HHV_VHV_mat(t, ham, coupling, coupling_args):
    """Build the extra controls of the form [H, [H, dlamH]] and [dlamH, [H, dlamH]]
    Parameters:
        ham (Hamiltonian_CD):       Counterdiabatic Hamiltonian of interest
        coupling (function):    Coupling function for the control term
        coupling_args (list):   Arguments for the coupling function
    """
    H = self.bareH.tocsr(time=t) if self.sparse else self.bareH.toarray(time=t)
    V = self.dlamH.tocsr(time=t) if self.sparse else self.dlamH.toarray(time=t)
    coeff = coupling(t, *coupling_args)
    HVcomm = H @ V - V @ H
    HHVmat = coeff * (H @ HVcomm - HVcomm @ H)
    VHVmat = coeff * (V @ HVcomm - HVcomm @ V)
    return HHVmat, VHVmat


def build_Hc1_Hc2_mat(ham, coupling, coupling_args):
    """Build
    Parameters:
        ham (Hamiltonian_CD):       Counterdiabatic Hamiltonian of interest
        coupling (function):    Coupling function for the control term
        coupling_args (list):   Arguments for the coupling function
    """
    H0 = self.H0.tocsr(time=t) if self.sparse else self.H0.toarray(time=t)
    H1 = self.H1.tocsr(time=t) if self.sparse else self.H1.toarray(time=t)
    coeff = coupling(t, *coupling_args)
    H1H0comm = H1 @ H0 - H0 @ H1
    Hc1mat = coeff * (H0 @ H1H0comm - H1H0comm @ H0)
    Hc2mat = coeff * (H1 @ H1H0comm - H1H0comm @ H1)
    return Hc1mat, Hc2mat


def build_controls_mat(ham, ctrl, coupling, coupling_args):
    """Build extra control terms from a list with different kinds of controls
    and the relevant couplings and parameters of those couplings
    Parameters:
        ham (Hamiltonian_CD):       Counterdiabatic Hamiltonian of interest
        ctrl (str):                 Type of control to add
        coupling (function):        Coupling function for control term
        coupling_args (list):       List of arguments for the coupling functions
    """
    if ctrl == "HHV_VHV":
        return build_HHV_VHV_mat(ham, coupling, coupling_args)
    elif ctrl == "Hc1_Hc2":
        return build_Hc1_Hc2_mat(ham, coupling, coupling_args)
    else:
        return build_controls_direct_mat(ham, ctrl, coupling, coupling_args)
