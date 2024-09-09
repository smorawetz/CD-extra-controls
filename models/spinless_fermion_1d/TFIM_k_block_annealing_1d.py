import os
import sys

import numpy as np
import quspin

sys.path.append(os.environ["CD_CODE_DIR"])

from .spinless_fermion_1d import Spinless_Fermion_1D
from tools.ham_couplings import turn_off_coupling, turn_on_coupling
from tools.connections import neighbours_1d


class TFIM_k_Block_Annealing_1D(Spinless_Fermion_1D):
    """Class for 1D transverse field Ising model single k block in momentum
    space after performing the Jordan-Wigner transformation."""

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

        J, hx, _ = map(lambda x: x * rescale, H_params)
        k = H_params[-1]
        onsite_J_coeff = -2 * J * np.cos(k)
        onsite_h_coeff = 2 * hx
        pair_coeff = -2 * J * np.sin(k)
        self.site0_J = [[onsite_J_coeff, 0, 0]]
        self.site0_h = [[onsite_h_coeff, 0, 0]]
        self.site1_J = [[onsite_J_coeff, 1, 1]]
        self.site1_h = [[onsite_h_coeff, 1, 1]]
        self.pp = [[pair_coeff, 0, 1]]
        self.mm = [[-pair_coeff, 0, 1]]
        self.shift = [[-hx, 0], [-hx, 1]]
        self.dlam_site0_h = [[-onsite_h_coeff, 0, 0]]
        self.dlam_site1_h = [[-onsite_h_coeff, 1, 1]]
        self.dlam_shift = [[hx, 0], [-hx, 1]]

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
            Nf=(0, 2),  # even # of particles, created/annihilated in pairs
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
        s = [["I", self.shift]]
        # s = []
        d = [
            ["+-", self.site0_J, turn_on_coupling, [self.schedule]],
            ["+-", self.site0_h, turn_off_coupling, [self.schedule]],
            ["+-", self.site1_J, turn_on_coupling, [self.schedule]],
            ["+-", self.site1_h, turn_off_coupling, [self.schedule]],
            ["++", self.pp, turn_on_coupling, [self.schedule]],
            ["--", self.mm, turn_on_coupling, [self.schedule]],
        ]
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_dlam_H(self):
        """Method for this particular spin model to calculate
        the $\lambda$ derivative of the Hamiltonian
        """
        s = [
            ["+-", self.site0_J],
            ["+-", self.dlam_site0_h],
            ["+-", self.site1_J],
            ["+-", self.dlam_site1_h],
            ["++", self.pp],
            ["--", self.mm],
            ["I", self.dlam_shift],
        ]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_target_H(self):
        """Method for this particular spin model to return the target
        Hamiltonian after annealing is complete, in the
        most symmetric possible basis to get the ground state easier
        """
        s = [
            ["+-", self.site0_J],
            ["+-", self.site1_J],
            ["++", self.pp],
            ["--", self.mm],
        ]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.targ_basis, **self.checks)
