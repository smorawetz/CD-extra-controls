import os
import sys

import quspin

sys.path.append(os.environ["CD_CODE_DIR"])

from .spinhalf_1d import SpinHalf_1D
from tools.ham_couplings import turn_off_coupling, turn_on_coupling
from tools.connections import neighbours_1d, next_neighbours_1d


class NNN_TFIM_Annealing_1D(SpinHalf_1D):
    """Class for 1D transverse field Ising model, with next-nearest-neighbor
    interactions as well, encoding local annealing problem between magnetic field
    polarized state at $\lambda = 0$ and ferromagnetic state at $\lambda = 1$"""

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

        J, J2, hx = map(lambda x: x * rescale, H_params)
        nn_pairs = neighbours_1d(Ns, boundary_conds)
        nnn_pairs = next_neighbours_1d(Ns, boundary_conds)
        self.J_terms = [[-J, *nn_pairs[i]] for i in range(len(nn_pairs))]
        self.J2_terms = [[-J2, *nnn_pairs[i]] for i in range(len(nnn_pairs))]
        self.hx_terms = [[-hx, i] for i in range(Ns)]
        self.flipped_hx_terms = [[hx, i] for i in range(Ns)]

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

    def build_H0(self):
        """Build QuSpin Hamiltonian for H0, which is component being
        "turned on" in annealing problem"""
        s = [["zz", self.J_terms], ["zz", self.J2_terms]]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_H1(self):
        """Method specific to this spin model to calculate
        the bare Hamiltonian (no controls or AGP)
        """
        s = [["x", self.hx_terms]]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_bare_H(self):
        """Method specific to this spin model to calculate
        the bare Hamiltonian (no controls or AGP)
        """
        s = []
        d = [
            ["zz", self.J_terms, turn_on_coupling, [self.schedule]],
            ["zz", self.J2_terms, turn_on_coupling, [self.schedule]],
            ["x", self.hx_terms, turn_off_coupling, [self.schedule]],
        ]
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_dlam_H(self):
        """Method for this particular spin model to calculate
        the $\lambda$ derivative of the Hamiltonian
        """
        s = [["zz", self.J_terms], ["zz", self.J2_terms], ["x", self.flipped_hx_terms]]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_target_H(self):
        """Method for this particular spin model to return the target
        Hamiltonian after annealing is complete, in the
        most symmetric possible basis to get the ground state easier
        """
        s = [["zz", self.J_terms], ["zz", self.J2_terms]]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.targ_basis, **self.checks)
