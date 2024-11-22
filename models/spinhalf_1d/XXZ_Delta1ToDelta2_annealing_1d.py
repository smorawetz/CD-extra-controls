import os
import sys

import quspin

sys.path.append(os.environ["CD_CODE_DIR"])

from .spinhalf_1d import SpinHalf_1D
from tools.ham_couplings import turn_off_coupling, turn_on_coupling
from tools.connections import neighbours_1d


class XXZ_Delta1ToDelta2_Annealing_1D(SpinHalf_1D):
    """Class for 1D XXZ model encoding local annealing
    problem between no anisotropy $\Delta = 0$ at $\lambda = 0$ and
    anisotropy dominated $\Delta -> \infty$ at $\lambda = 1$"""

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

        J, Delta = map(lambda x: x * rescale, H_params)
        pairs = neighbours_1d(Ns, boundary_conds)
        start_lam = 0.333
        end_lam = 0.5
        change_lam = end_lam - start_lam
        self.static_Delta_terms = [
            [-start_lam * Delta, *pairs[i]] for i in range(len(pairs))
        ]
        self.static_J_terms = [
            [-(1 - start_lam) * J / 4, *pairs[i]] for i in range(len(pairs))
        ]
        self.dynamic_Delta_terms = [
            [-change_lam * Delta, *pairs[i]] for i in range(len(pairs))
        ]
        self.dynamic_J_terms = [
            [change_lam * J / 4, *pairs[i]] for i in range(len(pairs))
        ]
        self.dlam_Delta_terms = [
            [-change_lam * Delta, *pairs[i]] for i in range(len(pairs))
        ]
        self.dlam_J_terms = [[change_lam * J / 4, *pairs[i]] for i in range(len(pairs))]
        self.final_Delta_terms = [
            [-end_lam * Delta, *pairs[i]] for i in range(len(pairs))
        ]
        self.final_J_terms = [
            [-(1 - end_lam) * J / 4, *pairs[i]] for i in range(len(pairs))
        ]

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
        s = [
            ["zz", self.static_Delta_terms],
            ["+-", self.static_J_terms],
            ["-+", self.static_J_terms],
        ]
        d = [
            ["zz", self.dynamic_Delta_terms, turn_on_coupling, [self.schedule]],
            ["+-", self.dynamic_J_terms, turn_on_coupling, [self.schedule]],
            ["-+", self.dynamic_J_terms, turn_on_coupling, [self.schedule]],
        ]
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_dlam_H(self):
        """Method for this particular spin model to calculate
        the $\lambda$ derivative of the Hamiltonian
        """
        s = [
            ["zz", self.dlam_Delta_terms],
            ["+-", self.dlam_J_terms],
            ["-+", self.dlam_J_terms],
        ]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_target_H(self):
        """Method for this particular spin model to return the target
        Hamiltonian after annealing is complete, in the
        most symmetric possible basis to get the ground state easier
        """
        s = [
            ["zz", self.final_Delta_terms],
            ["+-", self.final_J_terms],
            ["-+", self.final_J_terms],
        ]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.targ_basis, **self.checks)
