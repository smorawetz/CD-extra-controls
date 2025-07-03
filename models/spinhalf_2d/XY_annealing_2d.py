import os
import sys

import numpy as np
import quspin

sys.path.append(os.environ["CD_CODE_DIR"])

from .spinhalf_2d import SpinHalf_2D
from tools.ham_couplings import turn_off_coupling, turn_on_coupling
from tools.connections import neighbours_2d


class XY_Annealing_2D(SpinHalf_2D):
    """Class for 2D XY model encoding local annealing problem between
    Neel state at $lambda = 0$ and ferromagnetic state at $lambda = 1$"""

    def __init__(
        self,
        Nx,
        Ny,
        H_params,
        boundary_conds,
        agp_order,
        norm_type,
        schedule,
        symmetries={},
        target_symmetries={},
        rescale=1,
    ):

        omega, g = map(lambda x: x * rescale, H_params)
        Ns = Nx * Ny
        sites = np.arange(Ns)
        x = sites % Nx
        y = sites // Nx
        pairs = neighbours_2d(Nx, Ny, boundary_conds)
        self.omega_terms = [[omega * (-1) ** (x[i] + y[i]), i] for i in range(Ns)]
        self.flipped_omega_terms = [
            [-omega * (-1) ** (x[i] + y[i]), i] for i in range(Ns)
        ]
        self.g_terms = [[g / 4, *pairs[i]] for i in range(len(pairs))]

        super().__init__(
            Nx,
            Ny,
            H_params,
            boundary_conds,
            agp_order,
            norm_type,
            schedule,
            symmetries=symmetries,
            target_symmetries=target_symmetries,
            rescale=rescale,
        )

    def build_H0(self):
        """Build QuSpin Hamiltonian for H0, which is component being
        "turned on" in annealing problem"""
        s = [["+-", self.g_terms], ["-+", self.g_terms]]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_H1(self):
        """Method specific to this spin model to calculate
        the bare Hamiltonian (no controls or AGP)
        """
        s = [["z", self.omega_terms]]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_bare_H(self):
        """Method specific to this spin model to calculate
        the bare Hamiltonian (no controls or AGP)
        """
        s = []
        d = [
            ["+-", self.g_terms, turn_on_coupling, [self.schedule]],
            ["-+", self.g_terms, turn_on_coupling, [self.schedule]],
            ["z", self.omega_terms, turn_off_coupling, [self.schedule]],
        ]
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_dlam_H(self):
        """Method for this particular spin model to calculate
        the $lambda$ derivative of the Hamiltonian
        """
        s = [
            ["z", self.flipped_omega_terms],
            ["+-", self.g_terms],
            ["-+", self.g_terms],
        ]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_target_H(self):
        """Method for this particular spin model to return the target
        Hamiltonian after annealing is complete, in the
        most symmetric possible basis to get the ground state easier
        """
        s = [["+-", self.g_terms], ["-+", self.g_terms]]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.targ_basis, **self.checks)
