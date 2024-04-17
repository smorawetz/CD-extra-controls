import os
import sys

import numpy as np
import scipy
import quspin

sys.path.append(os.environ["CD_CODE_DIR"])

from base_hamiltonian import Base_Hamiltonian
from agp.krylov_construction import get_lanc_coeffs, get_gamma_vals
from agp.commutator_ansatz import get_alphas
from ham_controls.build_controls import build_controls_mat
from tools.lin_alg_calls import calc_comm


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

    def build_agp_mat_commutator(self, t, Hmat, dlamHmat):
        """Build matrix representing the AGP. This requires the atributes
        lanc_interp(t), gamma_interp(t), and alpha_interp(t) which give the
        Lanczos coefficients, the Krylov space AGP coefficients, and commutator
        ansatz AGP coefficients, respectively
        Parameters:
            t (float):                  Time at which to build the AGP term
            Hmat (np.array):            Matrix representation of the bare Hamiltonian
            dlamHmat (np.array):        Matrix representation of dlamH
        """
        alphas = self.alphas_interp(t)
        cmtr = calc_comm(Hmat, cmtr)
        AGPmat = 1j * alphas[0] * cmtr
        for n in range(1, self.agp_order):
            cmtr = calc_comm(Hmat, calc_comm(Hmat, cmtr))
            AGPmat += 1j * alphas[n] * cmtr
        return AGPmat  # TODO: confirm this is correct

    def build_agp_mat_krylov(self, t, Hmat, dlamHmat):
        """Build matrix representing the AGP. This requires the atributes
        lanc_interp(t), gamma_interp(t), and alpha_interp(t) which give the
        Lanczos coefficients, the Krylov space AGP coefficients, and commutator
        ansatz AGP coefficients, respectively
        Parameters:
            t (float):                  Time at which to build the AGP term
            Hmat (np.array):            Matrix representation of the bare Hamiltonian
            dlamHmat (np.array):        Matrix representation of dlamH
        """
        lanc_coeffs = self.lanc_interp(t)
        gammas = self.gammas_interp(t)
        O0 = dlamHmat
        O0 /= lanc_coeffs[0] + DIV_EPS
        O1 = calc_comm(Hmat, O0)
        O1 /= lanc_coeffs[1] + DIV_EPS
        lanc_coeffs.append(b1)
        AGPmat = 1j * gammas[0] * O1
        for n in range(1, agp_order):
            On = calc_comm(Hmat, O1) - lanc_coeffs[4 * n - 3] * O0
            On /= lanc_coeffs[4 * n - 2] + DIV_EPS
            O0 = O1
            O1 = On
            On = calc_comm(Hmat, O1) - lanc_coeffs[4 * n - 1] * O0
            On /= lanc_coeffs[4 * n] + DIV_EPS
            AGPmat += 1j * gammas[n] * On
            O0 = O1
            O1 = On
        return AGPmat

    def build_agp_mat(self, type, t, Hmat, dlamHmat):
        """Build matrix representing the AGP. This will give either the commutator
        or Lanczos expansion of the AGP, depending on the `type` parameter
        Parameters:
            type (str):                 Either "commutator" or "krylov"
            t (float):                  Time at which to build the AGP term
            Hmat (np.array):            Matrix representation of the bare Hamiltonian
            dlamHmat (np.array):        Matrix representation of dlamH
        """
        if type == "commutator":
            return self.build_agp_mat_commutator(t, Hmat, dlamHmat)
        elif type == "krylov":
            return self.build_agp_mat_krylov(t, Hmat, dlamHmat)
        else:
            raise ValueError("Invalid type for AGP construction")
