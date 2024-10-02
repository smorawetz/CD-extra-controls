import os
import sys

import numpy as np
import quspin

sys.path.append(os.environ["CD_CODE_DIR"])

from .spinless_fermion_1d import Spinless_Fermion_1D
from tools.ham_couplings import turn_off_coupling, turn_on_coupling
from tools.connections import neighbours_1d


class TFIM_Fermion_Annealing_1D(Spinless_Fermion_1D):
    """Class for 1D transverse field after Jordan-Wigner transformation
    to fermions, but remaining in real space"""

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
        rescale=1,
    ):

        self.Ns = Ns

        J, hx, _ = map(lambda x: x * rescale, H_params)
        Nd = H_params[2]

        np.random.seed(0)
        Jij_coeffs = [-J / 3 * np.random.rand() for _ in range(Nd + 1)]
        hi_coeffs = [hx / 3 * np.random.rand() for _ in range(Nd + 1)]

        Jij_coeffs = np.array(Jij_coeffs)
        hi_coeffs = np.array(hi_coeffs)
        ms = np.arange(len(hi_coeffs))

        self.J_pms = [
            [sum(Jij_coeffs * np.cos(2 * np.pi * i / 2**ms)), i, (i + 1)]
            for i in range(Ns - 1)
        ]
        self.J_mps = [
            [-sum(Jij_coeffs * np.cos(2 * np.pi * i / 2**ms)), i, (i + 1)]
            for i in range(Ns - 1)
        ]
        self.J_pps = [
            [sum(Jij_coeffs * np.cos(2 * np.pi * i / 2**ms)), i, (i + 1)]
            for i in range(Ns - 1)
        ]
        self.J_mms = [
            [-sum(Jij_coeffs * np.cos(2 * np.pi * i / 2**ms)), i, (i + 1)]
            for i in range(Ns - 1)
        ]
        self.h_pms = [
            [sum(2 * hi_coeffs * np.cos(2 * np.pi * i / 2**ms)), i]
            # [sum(hi_coeffs * np.cos(2 * np.pi * i / 2**ms)), i]
            for i in range(Ns)
        ]
        self.dlam_h_pms = [
            [-sum(2 * hi_coeffs * np.cos(2 * np.pi * i / 2**ms)), i]
            # [-sum(hi_coeffs * np.cos(2 * np.pi * i / 2**ms)), i]
            for i in range(Ns)
        ]

        # add final terms to close the chain with APBCs
        self.J_pms.append(
            [-sum(Jij_coeffs * np.cos(2 * np.pi * (Ns - 1) / 2**ms)), Ns - 1, 0]
        )
        self.J_mps.append(
            [sum(Jij_coeffs * np.cos(2 * np.pi * (Ns - 1) / 2**ms)), Ns - 1, 0]
        )
        self.J_pps.append(
            [-sum(Jij_coeffs * np.cos(2 * np.pi * (Ns - 1) / 2**ms)), Ns - 1, 0]
        )
        self.J_mms.append(
            [sum(Jij_coeffs * np.cos(2 * np.pi * (Ns - 1) / 2**ms)), Ns - 1, 0]
        )

        super().__init__(
            Ns,
            H_params,
            boundary_conds,
            agp_order,
            norm_type,
            schedule,
            symmetries=symmetries,
            target_symmetries=target_symmetries,
            rescale=rescale,
            Nf=range(0, Ns + 1, 2),
        )

    def build_H0(self):
        """Build QuSpin Hamiltonian for H0, which is component being
        "turned on" in annealing problem"""
        return None

    def build_H1(self):
        """Build QuSpin Hamiltonian for H1, which is component being
        "turned on" in annealing problem"""
        return None

    def build_bare_H(self):
        """Method specific to this spin model to calculate
        the bare Hamiltonian (no controls or AGP)
        """
        s = []
        d = [
            ["+-", self.J_pms, turn_on_coupling, [self.schedule]],
            ["-+", self.J_mps, turn_on_coupling, [self.schedule]],
            ["++", self.J_pps, turn_on_coupling, [self.schedule]],
            ["--", self.J_mms, turn_on_coupling, [self.schedule]],
            ["z", self.h_pms, turn_off_coupling, [self.schedule]],
        ]
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_dlam_H(self):
        """Method for this particular spin model to calculate
        the $\lambda$ derivative of the Hamiltonian
        """
        s = [
            ["+-", self.J_pms],
            ["-+", self.J_mps],
            ["++", self.J_pps],
            ["--", self.J_mms],
            ["z", self.dlam_h_pms],
        ]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_target_H(self):
        """Method for this particular spin model to return the target
        Hamiltonian after annealing is complete, in the
        most symmetric possible basis to get the ground state easier
        """
        s = [
            ["+-", self.J_pms],
            ["-+", self.J_mps],
            ["++", self.J_pps],
            ["--", self.J_mms],
        ]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.targ_basis, **self.checks)
