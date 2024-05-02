import os
import sys

import quspin

sys.path.append(os.environ["CD_CODE_DIR"])

from .spinhalf_1d import SpinHalf_1D
from tools.ham_couplings import turn_off_coupling, turn_on_coupling
from tools.connections import neighbours_1d


class Two_Spin_Test(SpinHalf_1D):
    """Class for simple two spin model, presented in Eq. (11) of
    PRL 123, 090602. Will be mostly used for testing simple cases"""

    def __init__(
        self,
        Ns,  # written for consistency, only sensible if = 2
        H_params,
        boundary_conds,  # doesn't matter
        agp_order,
        norm_type,
        schedule,
        symmetries={},
        target_symmetries={},
    ):

        J, hz = H_params
        pairs = [[0, 1]]
        self.J_terms = [[J, *pairs[i]] for i in range(len(pairs))]
        self.hz_terms = [[-hz, i] for i in range(Ns)]
        self.flipped_hz_terms = [[hz, i] for i in range(Ns)]

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

    # H0 and H1 not defined since this does not fit into that paradigm

    def build_bare_H(self):
        """Method specific to this spin model to calculate
        the bare Hamiltonian (no controls or AGP)
        """
        s = [["xx", self.J_terms], ["zz", self.J_terms]]
        d = [
            ["z", self.hz_terms, turn_off_coupling, [self.schedule]],
        ]
        return quspin.operators.hamiltonian(s, d, basis=self.basis)

    def build_dlam_H(self):
        """Method for this particular spin model to calculate
        the $\lambda$ derivative of the Hamiltonian
        """
        s = [["z", self.flipped_hz_terms]]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.basis)

    def build_target_H(self):
        """Method for this particular spin model to return the target
        Hamiltonian after annealing is complete, in the
        most symmetric possible basis to get the ground state easier
        """
        s = [["xy", self.J_terms], ["yx", self.J_terms]]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.target_basis)
