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

DIV_EPS = 1e-16


class Hamiltonian_CD(Base_Hamiltonian):
    """This is a "Hamiltonian" child class of the Base_Hamiltonian class,
    with the additional ability to construct a counterdiabatic term"""

    def __init__(
        self,
        Ns,
        H_params,
        boundary_conds,
        agp_order,
        norm_type,
        schedule,
        symmetries={},
        target_symmetries={},
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
            norm_type (str):            Either "trace" or "ground_state" for the norm
            schedule (Schedule):        Schedule object that encodes $\lambda(t)$
            symmetries (dictof (np.array, int)):    Symmetries of the Hamiltonian, which
                                        include a symmetry operation on the lattice
                                        and an integer which labels the sector by the
                                        eigenvalue of the symmetry transformation
            target_symmetries (dictof (np.array, int)):     Same as above, but for the
                                        target ground state if it has different symmetry
        """
        self.schedule = schedule
        self.agp_order = agp_order

        super().__init__(
            Ns,
            H_params,
            boundary_conds,
            symmetries=symmetries,
            target_symmetries=target_symmetries,
        )

    def calc_lanc_coeffs(self, t, norm_type, gstate=None):
        """Calculate the Lanczos coefficients for the for the action of the
        Liouvillian L = [H, .] on dlamH at a given time
        Parameters:
            t (float):          Time at which to calculate the Lanczos coefficients
            norm_type (str):    Either "trace" or "ground_state" for the norm
            gstate (np.array):  Ground state wavefunction to use in zero temp optimization
        """
        Hmat = self.bareH.tocsr(time=t) if self.sparse else self.bareH.toarray(time=t)
        O0 = self.dlamH.tocsr(time=t) if self.sparse else self.dlamH.toarray(time=t)
        return get_lanc_coeffs(
            self.agp_order, Hmat, O0, self.basis.Ns, norm_type, gstate=gstate
        )

    def calc_gammas(self, t):
        """Use the recursive solution for the coefficients of the Krylov space
        expansion of the AGP, denoted $\gamma_k$ in paper. Requires attribute
        lanc_interp(t) which gives all the Lanczos coefficients at time t
        Parameters:
            t (float):      Time at which to calculate the Lanczos coefficients
        """
        return get_gamma_vals(self.lanc_interp(t), self.agp_order)

    def calc_alphas(self, t, norm_type, gstate=None):
        """Use the matrix inverse solution for the coefficients of the commmutator
        expansion of the AGP, denoted $\alpha_k$ in paper.
        Parameters:
            t (float):          Time at which to calculate the Lanczos coefficients
            norm_type (str):    Either "trace" or "ground_state" for the norm
            gstate (np.array):  Ground state wavefunction to use in zero temp optimization
        """
        H = self.bareH.tocsr(time=t) if self.sparse else self.bareH.toarray(time=t)
        dlamH = self.dlamH.tocsr(time=t) if self.sparse else self.dlamH.toarray(time=t)
        return get_alphas(self.agp_order, H, dlamH, norm_type, gstate)

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
        cmtr = calc_comm(Hmat, dlamHmat)
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
        AGPmat = 1j * gammas[0] * O1
        for n in range(1, self.agp_order):
            On = calc_comm(Hmat, O1) - lanc_coeffs[2 * n - 1] * O0
            On /= lanc_coeffs[2 * n] + DIV_EPS
            O0 = O1
            O1 = On
            On = calc_comm(Hmat, O1) - lanc_coeffs[2 * n] * O0
            On /= lanc_coeffs[2 * n + 1] + DIV_EPS
            AGPmat += 1j * gammas[n] * On
            O0 = O1
            O1 = On
        return AGPmat

    def build_cd_term_mat(self, type, t, Hmat, dlamHmat):
        """Build matrix representing the AGP. This will give either the commutator
        or Lanczos expansion of the AGP, depending on the `type` parameter
        Parameters:
            type (str):                 Either "commutator" or "krylov"
            t (float):                  Time at which to build the AGP term
            Hmat (np.array):            Matrix representation of the bare Hamiltonian
            dlamHmat (np.array):        Matrix representation of dlamH
        """
        lamdot = self.schedule.get_lamdot(t)
        if type == "commutator":
            return lamdot * self.build_agp_mat_commutator(t, Hmat, dlamHmat)
        elif type == "krylov":
            return lamdot * self.build_agp_mat_krylov(t, Hmat, dlamHmat)
        else:
            raise ValueError("Invalid type for AGP construction")
