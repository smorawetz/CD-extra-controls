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
        symmetries=[],
        target_symmetries=[],
    ):
        """
        Parameters:
            Ns (int):                   Number of sites in lattice model (i.e. spins)
            H_params (listof float):    Parameters of the spin model,
                                        e.g. [1, 2] for J = 1 and h = 2
            boundary_cond (str):        Whether to use open ("open") or periodic
                                        ("periodic") boundary conditions. Defaults
                                        to open
            symmetries (dictof (np.array, int)):    Symmetries of the Hamiltonian, which
                                        include a symmetry operation on the lattice
                                        and an integer which labels the sector by the
                                        eigenvalue of the symmetry transformation
            target_symmetries (dictof (np.array, int)):     Same as above, but for the
                                        target ground state if it has different symmetry
        """

        self.Ns = Ns
        self.H_params = H_params
        self.boundary_conds = boundary_conds
        self.symmetries = symmetries
        self.target_symmetries = target_symmetries

    # should be able to have ground state, other functionality for bare, dlamH, etc.

    def build_H(self):
        """Build a QuSpin Hamiltonian object for the bare Hamiltonian
        Returns:
            H (quspin.operators.hamiltonian):           The bare Hamiltonian
        """
        return None

    def build_dlam_H(self):
        """Build a QuSpin Hamiltonian object for the $d_\lambda$ of bare Hamiltonian
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
        """Return the target (final) ground state encoded in the Hamiltonian
        Returns:
            target_gs (np.array):    The target ground state wavefunction
        """
        H = self.build_target_H()
        if H.basis.Ns >= 50:
            eigvals, eigvecs = H.eigsh(time=self.tau, k=1, which="SA")
        else:
            eigvals, eigvecs = H.eigh(time=self.tau)
        idx = eigvals.argsort()[0]
        target_gs = eigvecs[:, idx]
        if self.symmetries == self.target_symmetries:
            return target_gs
        else:
            return self.target_basis.project_from(target_gs, sparse=False)
