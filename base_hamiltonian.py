import numpy as np


class Base_Hamiltonian:
    """This is a base class to construct a "Hamiltonian", distinct from
    the class in QuSpin. This class has methods to build the full
    counterdiabatic Hamiltonian"""

    def __init__(
        self,
        Ns,
        H_params,
        boundary_conds,
    ):
        """
        Parameters:
            Ns (int):                   Number of sites in lattice model (i.e. spins)
            H_params (listof float):    Parameters of the spin model,
                                        e.g. [1, 2] for J = 1 and h = 2
            boundary_cond (str):        Whether to use open ("open") or periodic
                                        ("periodic") boundary conditions. Defaults
                                        to open
        """

        self.Ns= Ns
        self.H_params = H_params
        self.boundary_conds = boundary_conds

    # should be able to have ground state, other functionality for bare, dlamH, etc.

    def build_H(self):
        """Build a QuSpin Hamiltonian object for the bare Hamiltonian
        Returns:
            H (quspin.operators.hamiltonian):           The bare Hamiltonian
        """
        return None

    def build_dlam_H(self):
        """Build a QuSpin Hamiltonian object for the bare $d_\lambda$ Hamiltonian
        Returns:
            dlam_H (quspin.operators.hamiltonian):      $d_\lambda$ of bare Hamiltonian
        """
        return None

    def build_final_H(self):
        """Build a QuSpin Hamiltonian object for the Hamiltonian encoding the
        target ground state, including any symmetries of the final ground state
        Returns:
            final_H (quspin.operators.hamiltonian):     The bare Hamiltonian encoding the
                                                        target ground state, with
                                                        relevant symmetries included
        """
        return None

    def get_targ_gstate(self):
        """Return the final (target) ground state encoded in the Hamiltonian
        Returns:
            final_gs (np.array):    The final ground state wavefunction
        """
        H = self.build_final_H()
        if H.basis.Ns < 50:
            eigvals, eigvecs = H.eigsh(time=self.tau, k=1, which="SA")
        else:
            eigvals, eigvecs = H.eigh(time=self.tau)
        idx = eigvals.argsort()[0]
        final_gs = eigvecs[:, idx]
        return final_gs
        # if "symm" in self.model_name:
        #     return final_gs
        # else:
        #     return self.final_H_basis.project_from(final_gs, sparse=False)
