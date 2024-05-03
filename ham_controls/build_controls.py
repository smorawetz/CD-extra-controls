import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

from ham_controls.spin_half_controls import spin_half_controls
from tools.lin_alg_calls import calc_comm


def build_controls_direct(ham, ctrl, coupling, coupling_args):
    """Build the extra controls for the Hamiltonian, which are the
    counterdiabatic terms
    Parameters:
        ham (Hamiltonian_CD):   Counterdiabatic Hamiltonian of interest
        ctrl (str):             Type of control to add
        coupling (function):    Coupling function for the control term
        coupling_args (list):   Arguments for the coupling function
    """
    if ham.model_type in build_direct_dict.keys():
        control_term = build_direct_dict[ham.model_type](len(ctrl), ham.model_dim)
    else:
        raise ValueError("Need code for {0} direct controls".format(ham.model_type))
    return control_term(ham, ctrl, coupling, coupling_args)  # a quspin hamiltonian


def build_controls_direct_mat(t, ham, ctrl, coupling, coupling_args):
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


# TODO: time everything to see how fast this is, whether separately building arrays is too difficult
def build_HHV_mat(t, ham, coupling, coupling_args):
    """Build the extra controls of the form [H, [H, dlamH]] and [dlamH, [H, dlamH]]
    Parameters:
        ham (Hamiltonian_CD):       Counterdiabatic Hamiltonian of interest
        coupling (function):    Coupling function for the control term
        coupling_args (list):   Arguments for the coupling function
    """
    H = self.bareH.tocsr(time=t) if self.sparse else self.bareH.toarray(time=t)
    V = self.dlamH.tocsr(time=t) if self.sparse else self.dlamH.toarray(time=t)
    HVcomm = calc_comm(H, V)
    return coupling(t, *coupling_args) * calc_comm(H, HVcomm)


def build_VHV_mat(t, ham, coupling, coupling_args):
    """Build the extra controls of the form [H, [H, dlamH]] and [dlamH, [H, dlamH]]
    Parameters:
        ham (Hamiltonian_CD):       Counterdiabatic Hamiltonian of interest
        coupling (function):    Coupling function for the control term
        coupling_args (list):   Arguments for the coupling function
    """
    H = self.bareH.tocsr(time=t) if self.sparse else self.bareH.toarray(time=t)
    V = self.dlamH.tocsr(time=t) if self.sparse else self.dlamH.toarray(time=t)
    HVcomm = calc_comm(H, V)
    return coupling(t, *coupling_args) * calc_comm(V, HVcomm)


def build_Hc1_mat(ham, coupling, coupling_args):
    """Build
    Parameters:
        ham (Hamiltonian_CD):       Counterdiabatic Hamiltonian of interest
        coupling (function):    Coupling function for the control term
        coupling_args (list):   Arguments for the coupling function
    """
    H0 = self.H0.tocsr(time=t) if self.sparse else self.H0.toarray(time=t)
    H1 = self.H1.tocsr(time=t) if self.sparse else self.H1.toarray(time=t)
    H1H0comm = calc_comm(H1, H0)
    return coupling(t, *coupling_args) * calc_comm(H0, H1H0comm)


def build_Hc2_mat(ham, coupling, coupling_args):
    """Build
    Parameters:
        ham (Hamiltonian_CD):       Counterdiabatic Hamiltonian of interest
        coupling (function):    Coupling function for the control term
        coupling_args (list):   Arguments for the coupling function
    """
    H0 = self.H0.tocsr(time=t) if self.sparse else self.H0.toarray(time=t)
    H1 = self.H1.tocsr(time=t) if self.sparse else self.H1.toarray(time=t)
    H1H0comm = calc_comm(H1, H0)
    return coupling(t, *coupling_args) * calc_comm(H1, H0)


def build_controls_mat(t, ham, ctrl, coupling, coupling_args):
    """Build extra control terms from a list with different kinds of controls
    and the relevant couplings and parameters of those couplings
    Parameters:
        ham (Hamiltonian_CD):       Counterdiabatic Hamiltonian of interest
        ctrl (str):                 Type of control to add
        coupling (function):        Coupling function for control term
        coupling_args (list):       List of arguments for the coupling functions
    """
    if ctrl in build_mat_dict.keys():
        return build_mat_dict[ctrl](ham, coupling, coupling_args)
    else:
        return build_controls_direct_mat(t, ham, ctrl, coupling, coupling_args)


# dictionaries to access functions for building controls
build_direct_dict = {"spin-1/2": spin_half_controls}
build_mat_dict = {
    "HHV": build_HHV_mat,
    "VHV": build_VHV_mat,
    "Hc1": build_Hc1_mat,
    "Hc2": build_Hc2_mat,
}
