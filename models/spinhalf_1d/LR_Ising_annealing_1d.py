import os
import sys

import itertools
import quspin

sys.path.append(os.environ["CD_CODE_DIR"])

from .spinhalf_1d import SpinHalf_1D
from tools.ham_couplings import turn_off_coupling, turn_on_coupling
from tools.connections import neighbours_1d


class LR_Ising_Annealing_1D(SpinHalf_1D):
    """Class for 1D long-range Ising model encoding local annealing
    problem between magnetic field polarized state at $\lambda = 0$ and
    ferromagnetic state at $\lambda = 1$"""

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

        J, hx, alpha = H_params
        pairs = list(itertools.product(range(Ns), range(Ns)))
        for i in range(Ns):  # remove self-interaction
            pairs.remove((i, i))
        if boundary_conds == "periodic":
            dists = list(
                map(
                    lambda x: min(abs(x[1] - x[0]), Ns - abs(x[1] - x[0])),
                    pairs,
                )
            )
        elif boundary_conditions == "open":
            dists = list(map(lambda x: abs(x[1] - x[0]), pairs))
        self.J_terms = [[-J / dists[i] ** alpha, *pairs[i]] for i in range(len(pairs))]
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
        )

    def build_H0(self):
        """Build QuSpin Hamiltonian for H0, which is component being
        "turned on" in annealing problem"""
        s = [["zz", self.J_terms]]
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
            ["x", self.hx_terms, turn_off_coupling, [self.schedule]],
        ]
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_dlam_H(self):
        """Method for this particular spin model to calculate
        the $\lambda$ derivative of the Hamiltonian
        """
        s = [["zz", self.J_terms], ["x", self.flipped_hx_terms]]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.basis)

    def build_target_H(self):
        """Method for this particular spin model to return the target
        Hamiltonian after annealing is complete, in the
        most symmetric possible basis to get the ground state easier
        """
        s = [["zz", self.J_terms]]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.targ_basis, **self.checks)
