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
        rescale=1,
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
            rescale (float):            Rescale the Hamiltonian by this factor
        """

        self.Ns = Ns
        self.H_params = H_params
        self.boundary_conds = boundary_conds
        self.symmetries = symmetries
        self.target_symmetries = target_symmetries
        self.rescale = rescale

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
                self.targ_basis.project_from(target_gs, sparse=False), sparse=False
            )

    def get_bare_inst_gstate(self, t):
        """Return the instantaneous ground state of the BARE Hamiltonian at time t.
        This does NOT include the extra controls, if they are present.
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

    def get_spectral_function(self, t, ground_state=False, central_50=False):
        """Returns two arrays of points, corresponding to excitation frequencies
        $\omega$ and the value of the spectral function $\Phi(\omega)$ at these
        excitations, at a given instant of time
        Parameters:
            t (float):              Time at which to calculate the spectral function
            ground_state (bool):    Whether or not to return for the whole spectrum
                                    or just excitations connected to the ground state
        Returns:
            omegas (np.array):      The excitation frequencies
            phis (np.array):        The spectral function evaluated at omegas
        """
        E, V = self.bareH.eigh(time=t)
        inds = E.argsort()[::1]
        if central_50:
            inds = inds[int(0.25 * len(inds)) : int(0.75 * len(inds))]
        E = E[inds]
        V = V[:, inds]
        N = len(inds)
        dH_mat = self.dlamH.toarray(time=t)
        if ground_state:
            omegas = E[1:] - E[0]
            dHvec = np.matmul(np.conjugate(V[:, 0]), np.matmul(dH_mat, V[:, 1:]))
            phis = np.abs(dHvec) ** 2
        else:
            E_mat = E[np.repeat(np.arange(0, N), N).reshape(N, N)]
            E_mat = E_mat - E_mat.T  # gives Mij = Ei - Ej
            dH2_mat = np.matmul(np.conjugate(V.transpose()), np.matmul(dH_mat, V))
            dH2_mat = np.abs(dH2_mat) ** 2
            omegas = E_mat[np.tril_indices(len(E), -1)]
            phis = dH2_mat[np.tril_indices(len(E), -1)]
        return omegas, phis
