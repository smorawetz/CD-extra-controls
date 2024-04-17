import os
import sys

import quspin

sys.path.append(os.environ["CD_CODE_DIR"])

from .spinhalf_1d import SpinHalf_1D
from tools.ham_couplings import turn_off_coupling, turn_on_coupling


class TFIM_Annealing_1D(SpinHalf_1D):
    """Class for 1D transverse field Ising model encoding local annealing
    problem between magnetic field polarized state at $\lambda = 0$ and
    ferromagnetic state at $\lambda = 1$"""

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
        super().__init__(
            Ns,
            H_params,
            boundary_conds,
            agp_order,
            schedule,
            symmetries=symmetries,
            target_symmetries=target_symmetries,
            norm_type=norm_type,
        )

        J, hx = self.H_params
        self.J_terms = [[-J, *self.pairs[i]] for i in range(len(self.pairs))]
        self.hx_terms = [[-hx, i] for i in range(self.Ns)]
        self.flipped_hx_terms = [[hx, i] for i in range(self.Ns)]

    def build_H0(self):
        """Build QuSpin Hamiltonian for H0, which is component being
        "turned on" in annealing problem"""
        s = [["zz", self.J_terms]]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.basis)

    def build_H1(self):
        """Method specific to this spin model to calculate
        the bare Hamiltonian (no controls or AGP)
        """
        s = [["x", self.hx_terms]]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.basis)

    def build_bare_H(self):
        """Method specific to this spin model to calculate
        the bare Hamiltonian (no controls or AGP)
        """
        s = []
        d = [
            ["zz", self.J_terms, turn_on_coupling, [self.schedule]],
            ["x", self.hx_terms, turn_off_coupling, [self.schedule]],
        ]
        return quspin.operators.hamiltonian(s, d, basis=self.basis)

    def build_dlam_H(self):
        """Method for this particular spin model to calculate
        the $\lambda$ derivative of the Hamiltonian
        """
        s = [["zz", self.J_terms], ["x", self.flipped_hx_terms]]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.basis)

    def build_target_H_symmetric(self):
        """Method for this particular spin model to return the target
        Hamiltonian after annealing is complete, in the
        most symmetric possible basis to get the ground state easier
        """
        s = [["zz", self.J_terms]]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.target_basis)
