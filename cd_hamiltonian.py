import numpy as np
import scipy
import quspin

from base_hamiltonian import Base_Hamiltonian
from agp_utils.krylov_construction import get_lanc_coeffs, get_gamma_vals
from agp_utils.commutator_ansatz import get_alphas


class Hamiltonian_CD(Base_Hamiltonian):
    """This is a "Hamiltonian" child class of the Base_Hamiltonian class,
    with the additional ability to construct a counterdiabatic term"""

    def __init__(
        self,
        Ns,
        H_params,
        boundary_conds,
        agp_order,
        schedule,
        symmetries={},
        target_symmetries={},
        agp_orthog=True,
        norm_type="trace",
    ):
        """
        Parameters:
            Ns (int):                   Number of spins in spin model
            H_params (listof float):    Parameters of the spin model,
                                        e.g. [1, 2] for J = 1 and h = 2
            boundary_cond (str):        Whether to use open ("open") or periodic
                                        ("periodic") boundary conditions. Defaults
                                        to open
            agp_order (int):            Order of AGP ansatz
            schedule (Schedule):        Schedule object that encodes $\lambda(t)$
            symmetries (dictof (np.array, int)):    Symmetries of the Hamiltonian, which
                                        include a symmetry operation on the lattice
                                        and an integer which labels the sector by the
                                        eigenvalue of the symmetry transformation
            target_symmetries (dictof (np.array, int)):     Same as above, but for the
                                        target ground state if it has different symmetry
            agp_orthog (bool):          Whether or not to construct the AGP in the Krylov
                                        basis or via the commutator expansion ansatz. If
                                        True, uses the Krylov basis. If False, uses the
                                        regular commutator expansion ansatz
            norm_type (str):            What type of norm to use in the AGP. "trace" gives
                                        infinite temperature AGP, "ground_state" gives zero
                                        temperature
        """
        super().__init__(
            Ns,
            H_params,
            boundary_conds,
            symmetries=symmetries,
            target_symmetries=target_symmetries,
        )
        self.schedule = schedule
        self.agp_order = agp_order
        self.agp_orthog = agp_orthog
        self.norm_type = norm_type

    def calc_lanc_coeffs(self, t, gstate=None):
        """Calculate the Lanczos coefficients for the for the action of the
        Liouvillian L = [H, .] on dlamH at a given time
        Parameters:
            t (float):      Time at which to calculate the Lanczos coefficients
        """
        Hmat = self.bareH.tocsr(time=t) if self.sparse else self.bareH.toarray(time=t)
        O0 = self.dlamH.tocsr(time=t) if self.sparse else self.dlamH.toarray(time=t)
        return get_lanc_coeffs(Hmat, O0, gstate)

    def calc_gammas(self, t):
        """Use the recursive solution for the coefficients of the Krylov space
        expansion of the AGP, denoted $\gamma_k$ in paper. Requires attribute
        lanc_interp(t) which gives all the Lanczos coefficients at time t
        Parameters:
            t (float):      Time at which to calculate the Lanczos coefficients
        """
        return get_gamma_vals(self.lanc_interp(t), self.agp_order)

    def calc_alphas(self, t, gstate=None):
        """Use the matrix inverse solution for the coefficients of the commmutator
        expansion of the AGP, denoted $\alpha_k$ in paper.
        Parameters:
            t (float):      Time at which to calculate the Lanczos coefficients
        """
        H = self.bareH.tocsr(time=t) if self.sparse else self.bareH.toarray(time=t)
        dlamH = self.dlamH.tocsr(time=t) if self.sparse else self.dlamH.toarray(time=t)
        return get_alphas(self.agp_order, H, dlamH, self.norm_type, gstate)
