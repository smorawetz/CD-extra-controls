import os
import sys

import numpy as np
import quspin

sys.path.append(os.environ["CD_CODE_DIR"])

from .spinless_fermion_1d import Spinless_Fermion_1D
from tools.ham_couplings import turn_off_coupling, turn_on_coupling
from tools.connections import neighbours_1d


class TFIM_Cell_Random_k_Block(Spinless_Fermion_1D):
    """Class for 1D transverse field Ising model with uniform couplings and
    random magnetic fields (within a unit cell)"""

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

        J, hx, _, _, _ = map(lambda x: x * rescale, H_params)
        seed = H_params[2]
        full_Ns = H_params[3]
        k0 = H_params[4]

        self.Ns = Ns
        self.full_Ns = full_Ns

        Nc = Ns // 2

        # define coefficients within a unit cell
        np.random.seed(seed)
        # these are hi in real space
        real_hi_coeffs = hx * np.random.rand(Nc)
        hi_coeffs = np.fft.fft(real_hi_coeffs) / Nc  # normalization

        self.J_pms = []
        self.J_mps = []
        self.J_pps = []
        self.J_mms = []
        self.h_pms = []
        self.h_mps = []
        self.dlam_h_pms = []
        self.dlam_h_mps = []

        # start loop over states in k block with k0
        k = k0
        # find index in range [0, full_Ns - 1] for this state k0
        n = int(np.round(((full_Ns * k / np.pi - 1) / 2 + full_Ns / 2) % full_Ns))

        nvals = []
        kvals = []
        for _ in range(Nc):
            nvals.append(n)
            kvals.append(k)
            # need to determine what new jump is... do this by hand
            n = (n + full_Ns // Nc) % full_Ns
            nstar = n - full_Ns // 2 + 1
            k = (2 * nstar - 1) * np.pi / full_Ns

        # now want to reference these ks via indices within the k block by
        # referring to indices n in as [Ns // 2, Ns // 2 + 1, ..., Ns // 2 - 1]
        # to do this we create a dictionary

        ind_dict = {}
        for i, n in enumerate(nvals):
            ind_dict[n] = Ns // 2 + i
            ind_dict[self.flip_n(n)] = Ns // 2 - i - 1

        for i in range(len(nvals)):  # loop over positive k in block
            n = nvals[i]
            k = kvals[i]
            self.J_pms.append([-2 * J * np.cos(k), ind_dict[n], ind_dict[n]])
            self.J_pms.append(
                [-2 * J * np.cos(k), ind_dict[self.flip_n(n)], ind_dict[self.flip_n(n)]]
            )
            self.J_pps.append(
                [-2 * J * np.sin(k), ind_dict[n], ind_dict[self.flip_n(n)]]
            )
            self.J_mms.append(
                [2 * J * np.sin(k), ind_dict[n], ind_dict[self.flip_n(n)]]
            )
            for m in np.arange(Nc):
                n_step = full_Ns // Nc * m
                # disorder causes coupling between different k modes
                self.h_pms.append(
                    [
                        # hi_coeffs[m],
                        np.conj(hi_coeffs[m]),
                        ind_dict[self.add_n(n, n_step)],
                        ind_dict[n],
                    ]
                )
                self.h_pms.append(
                    [
                        hi_coeffs[m],
                        # np.conj(hi_coeffs[m]),
                        ind_dict[n],
                        ind_dict[self.add_n(n, n_step)],
                    ]
                )
                self.h_pms.append(
                    [
                        hi_coeffs[m],
                        # np.conj(hi_coeffs[m]),
                        ind_dict[self.flip_n(self.add_n(n, n_step))],
                        ind_dict[self.flip_n(n)],
                    ]
                )
                self.h_pms.append(
                    [
                        # hi_coeffs[m],
                        np.conj(hi_coeffs[m]),
                        ind_dict[self.flip_n(n)],
                        ind_dict[self.flip_n(self.add_n(n, n_step))],
                    ]
                )
                #############################
                self.dlam_h_pms.append(
                    [
                        # -hi_coeffs[m],
                        -np.conj(hi_coeffs[m]),
                        ind_dict[self.add_n(n, n_step)],
                        ind_dict[n],
                    ]
                )
                self.dlam_h_pms.append(
                    [
                        -hi_coeffs[m],
                        # -np.conj(hi_coeffs[m]),
                        ind_dict[n],
                        ind_dict[self.add_n(n, n_step)],
                    ]
                )
                self.dlam_h_pms.append(
                    [
                        -hi_coeffs[m],
                        # -np.conj(hi_coeffs[m]),
                        ind_dict[self.flip_n(self.add_n(n, n_step))],
                        ind_dict[self.flip_n(n)],
                    ]
                )
                self.dlam_h_pms.append(
                    [
                        # -hi_coeffs[m],
                        -np.conj(hi_coeffs[m]),
                        ind_dict[self.flip_n(n)],
                        ind_dict[self.flip_n(self.add_n(n, n_step))],
                    ]
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

    def flip_n(self, n):
        return self.full_Ns - n - 1

    def add_n(self, n, kick):
        return (n + kick) % self.full_Ns

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
