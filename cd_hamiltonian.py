import os
import sys

import numpy as np
import scipy
import quspin

sys.path.append(os.environ["CD_CODE_DIR"])

from base_hamiltonian import Base_Hamiltonian
from agp.krylov_construction import get_lanc_coeffs, get_gamma_vals
from ham_controls.build_controls_ham import (
    build_H_controls_mat,
    build_dlamH_controls_mat,
)
from agp.commutator_ansatz import get_alphas
from ham_controls.build_controls import build_controls_direct, build_mat_dict
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
            Ns (int):                       Number of spins in spin model
            H_params (listof float):        Parameters of the spin model,
                                            e.g. [1, 2] for J = 1 and h = 2
            boundary_cond (str):            Whether to use open ("open") or periodic
                                            ("periodic") boundary conditions. Defaults
                                            to open
            agp_order (int):                Order of AGP ansatz
            norm_type (str):                Either "trace" or "ground_state" for the norm
            schedule (Schedule):            Schedule object that encodes $\lambda(t)$
            symmetries (dictof (np.array, int)):    Symmetries of the Hamiltonian, which
                                            include a symmetry operation on the lattice
                                            and an integer which labels the sector by the
                                            eigenvalue of the symmetry transformation
            target_symmetries (dictof (np.array, int)):     Same as above, but for the
                                            target ground state if it has different symmetry
        """
        self.agp_order = agp_order
        self.schedule = schedule
        self.ctrls = []
        self.ctrls_couplings = []
        self.ctrls_args = []

        self.checks = {"check_herm": False, "check_symm": False}

        super().__init__(
            Ns,
            H_params,
            boundary_conds,
            symmetries=symmetries,
            target_symmetries=target_symmetries,
        )

    def init_controls(self, ctrls, ctrls_couplings, ctrls_args):
        """Add extra control terms to the CD Hamiltonian object
        Parameters:
            ctrls (listof str):                 List of control types to add
            ctrls_couplings (listof str):       List of strings corresponding to coupling
                                                functions of control terms
            ctrls_args (listof list):           List of list of arguments for the coupling functions
        """
        self.ctrls = ctrls
        self.ctrls_couplings = ctrls_couplings
        self.ctrls_args = ctrls_args

    def calc_lanc_coeffs(self, t, norm_type, gstate=None):
        """Calculate the Lanczos coefficients for the for the action of the
        Liouvillian L = [H, .] on dlamH at a given time
        Parameters:
            t (float):          Time at which to calculate the Lanczos coefficients
            norm_type (str):    Either "trace" or "ground_state" for the norm
            gstate (np.array):  Ground state wavefunction to use in zero temp optimization
        """
        Hmat = build_H_controls_mat(
            self, t, self.ctrls, self.ctrls_couplings, self.ctrls_args
        )
        O0 = build_dlamH_controls_mat(
            self, t, self.ctrls, self.ctrls_couplings, self.ctrls_args
        )
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
        Hmat = build_H_controls_mat(
            self, t, self.ctrls, self.ctrls_couplings, self.ctrls_args
        )
        dlamH = build_dlamH_controls_mat(
            self, t, self.ctrls, self.ctrls_couplings, self.ctrls_args
        )
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
        O0 = dlamHmat.copy()
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

    def build_agp_mat(self, t, AGPtype, Hmat, dlamHmat):
        """Build matrix representing the AGP. This will give either the commutator
        or Lanczos expansion of the AGP, depending on the `AGPtype` parameter
        Parameters:
            t (float):                  Time at which to build the AGP term
            AGPtype (str):              Either "commutator" or "krylov"
            Hmat (np.array):            Matrix representation of the bare Hamiltonian
            dlamHmat (np.array):        Matrix representation of dlamH
        """
        if AGPtype == "commutator":
            return self.build_agp_mat_commutator(t, Hmat, dlamHmat)
        elif AGPtype == "krylov":
            return self.build_agp_mat_krylov(t, Hmat, dlamHmat)
        else:
            raise ValueError("Invalid type for AGP construction")

    def build_cd_term_mat(self, t, AGPtype, Hmat, dlamHmat):
        """Build matrix representing the counterdiabatic term, which is just
        $\dot{\lambda}$ multiplied by the AGP. This will give either the commutator
        or Lanczos construction of the AGP, depending on the `AGPtype` parameter
        Parameters:
            t (float):                  Time at which to build the AGP term
            AGPtype (str):              Either "commutator" or "krylov"
            Hmat (np.array):            Matrix representation of the bare Hamiltonian
            dlamHmat (np.array):        Matrix representation of dlamH
        """
        lamdot = self.schedule.get_lamdot(t)
        return lamdot * self.build_agp_mat(t, AGPtype, Hmat, dlamHmat)

    def build_cons_G_mat(self, t, AGPtype, Hmat, dlamHmat):
        """Build the matrix representing the (approximately) conserved quantity
        $G_\lambda$, whose off-diagonal elements (in the energy eigenbasis) are
        minimized to obtain the approxiamte AGP
        Parameters:
            t (float):                  Time at which to build the AGP term
            AGPtype (str):              Either "commutator" or "krylov"
            Hmat (np.array):            Matrix representation of the bare Hamiltonian
            dlamHmat (np.array):        Matrix representation of dlamH
        """
        Alam = self.build_agp_mat(t, AGPtype, Hmat, dlamHmat)
        return dlamHmat + 1j * calc_comm(Alam, Hmat)
