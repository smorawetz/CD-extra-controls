import os
import sys

import quspin

sys.path.append(os.environ["CD_CODE_DIR"])

from .spinhalf_1d import SpinHalf_1D
from tools.ham_couplings import turn_off_coupling, turn_on_coupling
from tools.connections import neighbours_1d


class XXZ_HalfHeisenberg_Annealing_1D(SpinHalf_1D):
    """Class for 1D XXZ model encoding local annealing
    problem between Heisenberg point with $J = \Delta$ at $\lambda = 0$ and
    anisotropy dominated $\Delta / J -> \infty$ at $\lambda = 1$"""

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

        J, _ = map(lambda x: x * rescale, H_params)
        pairs = neighbours_1d(Ns, boundary_conds)
        self.pm_terms = [[-J / 5, *pairs[i]] for i in range(len(pairs))]
        self.flipped_pm_terms = [[J / 5, *pairs[i]] for i in range(len(pairs))]
        self.off_zz_terms = [[-J / 5, *pairs[i]] for i in range(len(pairs))]
        self.flipped_off_zz_terms = [[J / 5, *pairs[i]] for i in range(len(pairs))]
        self.on_zz_terms = [[-J, *pairs[i]] for i in range(len(pairs))]

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
        s = [["+-", self.pm_terms], ["-+", self.pm_terms], ["zz", self.off_zz_terms]]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_H1(self):
        """Method specific to this spin model to calculate
        the bare Hamiltonian (no controls or AGP)
        """
        s = [["zz", self.on_zz_terms]]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_bare_H(self):
        """Method specific to this spin model to calculate
        the bare Hamiltonian (no controls or AGP)
        """
        s = []
        d = [
            ["zz", self.on_zz_terms, turn_on_coupling, [self.schedule]],
            ["zz", self.off_zz_terms, turn_off_coupling, [self.schedule]],
            ["+-", self.pm_terms, turn_off_coupling, [self.schedule]],
            ["-+", self.pm_terms, turn_off_coupling, [self.schedule]],
        ]
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_dlam_H(self):
        """Method for this particular spin model to calculate
        the $\lambda$ derivative of the Hamiltonian
        """
        s = [
            ["zz", self.on_zz_terms],
            ["zz", self.flipped_off_zz_terms],
            ["+-", self.flipped_pm_terms],
            ["-+", self.flipped_pm_terms],
        ]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_target_H(self):
        """Method for this particular spin model to return the target
        Hamiltonian after annealing is complete, in the
        most symmetric possible basis to get the ground state easier
        """
        s = [["zz", self.on_zz_terms]]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.targ_basis, **self.checks)
