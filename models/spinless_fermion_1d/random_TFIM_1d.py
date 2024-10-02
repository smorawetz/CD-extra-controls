import os
import sys

import numpy as np
import quspin

sys.path.append(os.environ["CD_CODE_DIR"])

from .spinless_fermion_1d import Spinless_Fermion_1D
from tools.ham_couplings import turn_off_coupling, turn_on_coupling
from tools.connections import neighbours_1d


class TFIM_k_Block_Random_1D(Spinless_Fermion_1D):
    """Class for 1D transverse field Ising model with (periodic) random couplings and
    magnetic fields (all with same sign for both)."""

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

        J, hx, _, _, _ = map(lambda x: x * rescale, H_params)
        Nd = H_params[2]
        full_Ns = H_params[3]
        k0 = H_params[4]

        ## add seed somewhere else
        np.random.seed(0)
        Jij_coeffs = [-J / 3 * np.random.rand() for _ in range(Nd + 1)]
        hi_coeffs = [hx / 3 * np.random.rand() for _ in range(Nd + 1)]

        self.J_pms = []
        self.J_mps = []
        self.J_pps = []
        self.J_mms = []
        self.h_pms = []
        self.h_mps = []
        self.dlam_h_pms = []
        self.dlam_h_mps = []

        k_ind = Ns // 2
        i = 0
        while i < Ns // 2:  # loop over positive k in block
            k = k0
            for m in np.arange(Nd + 1):
                # group 3
                self.J_pms.append(
                    [
                        Jij_coeffs[m]
                        * np.cos(k + np.pi / 2**m)
                        * np.exp(-1j * np.pi / 2**m),
                        self.add_k_ind(k_ind, Ns // 2**m),
                        k_ind,
                    ]
                )
                self.J_pms.append(
                    [
                        Jij_coeffs[m]
                        * np.cos(k + np.pi / 2**m)
                        * np.exp(1j * np.pi / 2**m),
                        k_ind,
                        self.add_k_ind(k_ind, Ns // 2**m),
                    ]
                )
                # group 4
                self.J_pms.append(
                    [
                        Jij_coeffs[m]
                        * np.cos(k + np.pi / 2**m)
                        * np.exp(1j * np.pi / 2**m),
                        self.flip_k_ind(self.add_k_ind(k_ind, Ns // 2**m)),
                        self.flip_k_ind(k_ind),
                    ]
                )
                self.J_pms.append(
                    [
                        Jij_coeffs[m]
                        * np.cos(k + np.pi / 2**m)
                        * np.exp(-1j * np.pi / 2**m),
                        self.flip_k_ind(k_ind),
                        self.flip_k_ind(self.add_k_ind(k_ind, Ns // 2**m)),
                    ]
                )
                # group 5
                self.h_pms.append(
                    [hi_coeffs[m], self.add_k_ind(k_ind, Ns // 2**m), k_ind]
                )
                self.h_pms.append(
                    [hi_coeffs[m], k_ind, self.add_k_ind(k_ind, Ns // 2**m)]
                )
                self.h_pms.append(
                    [
                        hi_coeffs[m],
                        self.flip_k_ind(self.add_k_ind(k_ind, Ns // 2**m)),
                        self.flip_k_ind(k_ind),
                    ]
                )
                self.h_pms.append(
                    [
                        hi_coeffs[m],
                        self.flip_k_ind(k_ind),
                        self.flip_k_ind(self.add_k_ind(k_ind, Ns // 2**m)),
                    ]
                )
                #############################
                self.dlam_h_pms.append(
                    [-hi_coeffs[m], self.add_k_ind(k_ind, Ns // 2**m), k_ind]
                )
                self.dlam_h_pms.append(
                    [-hi_coeffs[m], k_ind, self.add_k_ind(k_ind, Ns // 2**m)]
                )
                self.dlam_h_pms.append(
                    [
                        -hi_coeffs[m],
                        self.flip_k_ind(self.add_k_ind(k_ind, Ns // 2**m)),
                        self.flip_k_ind(k_ind),
                    ]
                )
                self.dlam_h_pms.append(
                    [
                        -hi_coeffs[m],
                        self.flip_k_ind(k_ind),
                        self.flip_k_ind(self.add_k_ind(k_ind, Ns // 2**m)),
                    ]
                )
                # group 1
                self.J_pps.append(
                    [
                        -Jij_coeffs[m]
                        * np.sin(k + np.pi / 2**m)
                        * np.exp(-1j * np.pi / 2**m),
                        self.flip_k_ind(k_ind),
                        self.add_k_ind(k_ind, Ns // 2**m),
                    ]
                )
                self.J_pps.append(
                    [
                        -Jij_coeffs[m]
                        * np.sin(k + np.pi / 2**m)
                        * np.exp(1j * np.pi / 2**m),
                        self.flip_k_ind(self.add_k_ind(k_ind, Ns // 2**m)),
                        k_ind,
                    ]
                )
                # group 2
                self.J_mms.append(
                    [
                        -Jij_coeffs[m]
                        * np.sin(k + np.pi / 2**m)
                        * np.exp(1j * np.pi / 2**m),
                        self.add_k_ind(k_ind, Ns // 2**m),
                        self.flip_k_ind(k_ind),
                    ]
                )
                self.J_mms.append(
                    [
                        -Jij_coeffs[m]
                        * np.sin(k + np.pi / 2**m)
                        * np.exp(-1j * np.pi / 2**m),
                        k_ind,
                        self.flip_k_ind(self.add_k_ind(k_ind, Ns // 2**m)),
                    ]
                )
            n = ((full_Ns * k0 / np.pi - 1) / 2 + full_Ns / 2) % full_Ns
            nstar = (n + full_Ns / 2**Nd) % full_Ns
            k_ind = int(nstar)
            nstar = nstar - full_Ns / 2 + 1
            k0 = (2 * nstar - 1) * np.pi / full_Ns
            i += 1

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

    def flip_k_ind(self, k_ind):
        return self.Ns - k_ind - 1

    def add_k_ind(self, k_ind, kick):
        return (k_ind + kick) % self.Ns

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
            ["+-", self.h_pms, turn_off_coupling, [self.schedule]],
            ["-+", self.h_mps, turn_off_coupling, [self.schedule]],
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
            ["+-", self.dlam_h_pms],
            ["-+", self.dlam_h_mps],
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
