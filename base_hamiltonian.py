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
        symmetries={},
        target_symmetries={},
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

        self.bareH = self.build_bare_H()
        self.dlamH = self.build_dlam_H()
        self.H0 = self.build_H0()
        self.H1 = self.build_H1()

        self.sparse = True if self.bareH.basis.Ns >= 40 else False

    # should be able to have ground state, other functionality for bare, dlamH, etc.

    def build_bare_H(self):
        """Build a QuSpin Hamiltonian object for the bare Hamiltonian"""
        return None

    def build_dlam_H(self):
        """Build a QuSpin Hamiltonian object for the $d_\lambda$ of bare Hamiltonian"""
        return None

    def build_final_H(self):
        """Build a QuSpin Hamiltonian object for the Hamiltonian encoding the
        target ground state, including any symmetries of the final ground state"""
        return None

    def build_H0(self):
        """Build QuSpin Hamiltonian for H0, which is component being
        "turned on" in annealing problem"""
        return None

    def build_H1(self):
        """Method specific to this spin model to calculate
        the bare Hamiltonian (no controls or AGP)
        """
        return None

    def get_init_gstate(self):
        """Return the initial ground state encoded in the Hamiltonian
        Returns:
            init_gs (np.array):    The target ground state wavefunction
        """
        if self.sparse:
            eigvals, eigvecs = self.bareH.eigsh(time=0, k=1, which="SA")
        else:
            eigvals, eigvecs = self.bareH.eigh(time=0)
        idx = eigvals.argsort()[0]
        init_gs = eigvecs[:, idx]
        return init_gs

    def get_targ_gstate(self):
        """Return the target (final) ground state encoded in the Hamiltonian
        Returns:
            target_gs (np.array):    The target ground state wavefunction
        """
        H = self.build_target_H()
        if self.sparse:
            eigvals, eigvecs = H.eigsh(time=self.schedule.tau, k=1, which="SA")
        else:
            eigvals, eigvecs = H.eigh(time=self.schedule.tau)
        idx = eigvals.argsort()[0]
        target_gs = eigvecs[:, idx]
        if self.symmetries == self.target_symmetries:
            return target_gs
        else:
            return self.basis.project_to(
                self.target_basis.project_from(target_gs), sparse=False
            )

    def get_inst_gstate(self, t):
        """Return the instantaneous ground state of the Hamiltonian at time t
        Parameters:
            t (float):      Time at which to calculate the ground state
        Returns:
            inst_gs (np.array):     The instantaneous ground state wavefunction
        """
        if self.sparse:
            eigvals, eigvecs = self.bareH.eigsh(time=t, k=1, which="SA")
        else:
            eigvals, eigvecs = self.bareH.eigh(time=t)
        idx = eigvals.argsort()[0]
        inst_gs = eigvecs[:, idx]
        return inst_gs

    def build_H_mats(self, t, ctrls, couplings, coupling_args):
        """Returns a list of (sparse or dense) matrices, which are added to
        form H. These may be used in matrix-vector multiplication on the wavefunction
        Parameters:
            t (float):                      Time at which to build the Hamiltonian
            ctrls (listof str):             List of control types to add
            couplings (listof function):    List of dlam_coupling functions for
                                            control terms
            coupling_args (listof list):    List of arguments for the coupling functions
        """
        # need to get bare H, controls, and AGP term
        bareH = self.bareH.tocsr(time=t) if self.sparse else self.bareH.toarray(time=t)
        Hmats = [bareH]
        for i in range(len(ctrls)):
            Hmats.append(
                build_controls_mat(t, self, ctrls[i], couplings[i], couplings_args[i])
            )
        return Hmats
