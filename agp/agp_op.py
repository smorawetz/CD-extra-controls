from utils.commutator_ansatz import compute_alphas


class AGP:
    """This class constructs all the relevant parts of the approximate
    AGP for a given counterdiabatic Hamiltonian"""

    def __init__(
        self, bare_H, agp_order, agp_orthog=True, norm_type="trace", interp_points=300
    ):
        """
        Parameters:
            bare_H (CD_Hamiltonian):    The bare Hamiltonian of the system
                                        for which the AGP is to be calculated
            agp_order (int):            Order of AGP ansatz
            agp_orthog (bool):          Whether or not to construct the AGP in the Krylov
                                        basis or via the commutator expansion ansatz. If
                                        True, uses the Krylov basis. If False, uses the
                                        regular commutator expansion ansatz
            norm_type (str):            What type of norm to use in the AGP. "trace" gives
                                        infinite temperature AGP, "ground_state" gives zero
                                        temperature
            interp_points (int):        Number of points to build in interpolation grid for
                                        AGP coefficients
        """
        self.bare_H = bare_H
        self.agp_order = agp_order
        self.agp_orthog = agp_orthog
        self.norm_type = norm_type

    def commutator_grid(self, sched):
        """Compute the AGP coefficients in the commutator ansatz on a
        grid of different times"""
        quspin_H = self.bare_H.build_H()
        quspin_dlam_H = self.bare_H.build_dlam_H()

        lam_grid = np.linspace(0, 1, self.interp_points)
        alphas_grid = np.zeros((self.agp_order, self.interp_points))
        for i in range(self.interp_points):
            lam = lam_grid[i]
            t = sched.get_t(lam)

            if quspin_H.Ns < 50:  # for small systems, dense matrices are faster
                H = quspin_H.todense(time=t)
                dlam_H = quspin_dlam_H.todense(time=t)
            else:
                H = quspin_H.tocsr(time=t)
                dlam_H = quspin_dlam_H.tocsr(time=t)

            if norm_type == "ground_state":
                gstate = quspin_H.get_gstate()

            alphas_grid[:, i] = compute_alphas(
                self.agp_order, H, dlam_H, self.norm_type
            )
