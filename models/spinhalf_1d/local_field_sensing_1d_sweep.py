import os
import sys

import numpy as np
import quspin

sys.path.append(os.environ["CD_CODE_DIR"])

from .spinhalf_1d import SpinHalf_1D
from tools.ham_couplings import sweep_sign, off_on_off_sweep, dlam_off_on_off_sweep
from tools.connections import neighbours_1d


class Local_Field_Sensing_1D_Sweep(SpinHalf_1D):
    """Class for a localling interacting spin chain, with an annealing
    transverse field and a symmetry-breaking longitudinal field, whose
    e.g. strength is detected by measuring the state. The protcol sweeps
    from a strong transverse field in one direction to the opposite"""

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
        J, hx, hz, disorder_strength, _ = map(lambda x: x * rescale, H_params)
        disorder_seed = H_params[-1]

        np.random.seed(disorder_seed)
        on_site_disorder = np.random.uniform(-disorder_strength, disorder_strength, Ns)

        pairs = neighbours_1d(Ns, boundary_conds)
        self.J_terms = [[-J, *pairs[i]] for i in range(len(pairs))]
        self.hx_terms = [[hx, i] for i in range(Ns)]
        self.double_hx_terms = [[2 * hx, i] for i in range(Ns)]
        self.hz_terms = [[-hz, i] for i in range(Ns)]
        self.disorder_terms = [[on_site_disorder[i], i] for i in range(Ns)]

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

    # does not fit into traditional H_0, H_1 structure
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
        s = [["z", self.hz_terms], ["z", self.disorder_terms]]
        d = [
            ["zz", self.J_terms, off_on_off_sweep, [self.schedule]],
            ["x", self.hx_terms, sweep_sign, [self.schedule]],
        ]
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_dlam_H(self):
        """Method for this particular spin model to calculate
        the $\lambda$ derivative of the Hamiltonian
        """
        s = [["x", self.double_hx_terms]]
        d = [["zz", self.J_terms, dlam_off_on_off_sweep, [self.schedule]]]
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_target_H(self):
        """Method for this particular spin model to return the target
        Hamiltonian after annealing is complete, in the
        most symmetric possible basis to get the ground state easier
        """
        s = [["z", self.hz_terms], ["x", self.hx_terms]]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.targ_basis, **self.checks)
