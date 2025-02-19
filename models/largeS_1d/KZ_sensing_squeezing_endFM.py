import os
import sys

import numpy as np
import quspin

sys.path.append(os.environ["CD_CODE_DIR"])

from .largeS_1d import LargeS_1D
from tools.ham_couplings import sweep_sign, turn_on_coupling, turn_off_coupling
from tools.connections import neighbours_1d


class KZ_Sensing_Spin_Squeezing_EndFM(LargeS_1D):
    """Class for a large-S spin, with a single particle. There is a nonlinear
    interaction term (corresponding to Ising interactions in infinite-dimensional
    Ising chain) and a transverse field which is swept from twice the nonlinear term
    strength across the QCP to zero
    """

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
        chi, h, g = map(lambda x: x * rescale, H_params)
        S = Ns / 2
        chi = chi / S  # rescale so H is extensive in S

        self.chi_terms = [[-chi / 2, 0, 0]]  # divide by 2 so QCP at h = chi
        self.h_terms = [[-h / 2, 0]]  # div by 2 for +, -
        self.g_terms = [[g, 0]]
        self.double_g_terms = [[2 * g, 0]]

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
        )

    # does not fit into traditional H_0, H_1 structure
    def build_H0(self):
        """Build QuSpin Hamiltonian for H0, which is component being
        "turned on" in annealing problem"""
        return None

    def build_H1(self):
        """Method specific to this spin model to calculate
        the bare Hamiltonian (no controls or AGP)
        """
        return None

    def build_bare_H(self):
        """Method specific to this spin model to calculate
        the bare Hamiltonian (no controls or AGP)
        """
        s = [["+", self.h_terms], ["-", self.h_terms], ["zz", self.chi_terms]]
        d = [["z", self.g_terms, sweep_sign, [self.schedule]]]
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_dlam_H(self):
        """Method for this particular spin model to calculate
        the $\lambda$ derivative of the Hamiltonian
        """
        s = [["z", self.double_g_terms]]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_target_H(self):
        """Method for this particular spin model to return the target
        Hamiltonian after annealing is complete, in the
        most symmetric possible basis to get the ground state easier
        """
        s = [
            ["+", self.h_terms],
            ["-", self.h_terms],
            ["zz", self.chi_terms],
            ["z", self.g_terms],
        ]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.targ_basis, **self.checks)
