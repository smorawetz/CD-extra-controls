import os
import sys

import numpy as np
import quspin

sys.path.append(os.environ["CD_CODE_DIR"])

from .spinhalf_2d import SpinHalf_2D
from tools.ham_couplings import sweep_sign
from tools.connections import neighbours_2d


class Disorder_Ising_2D(SpinHalf_2D):
    """Class representing Ising models on a ladder with multiple kinds of disorder:
    random choice of the couplings (in sign), as well as on site disorder"""

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
        # model-specific kwargs
        J_disorder_type="none",
        onsite_disorder_type="uniform",
    ):
        J, hx, site_disord, _ = map(lambda x: x * rescale, H_params)
        disorder_seed = H_params[-1]

        np.random.seed(disorder_seed)
        Ns = Nx * Ny
        pairs = neighbours_2d(Nx, Ny, boundary_conds)
        if J_disorder_type == "none":
            J_couplings = np.ones(len(pairs)) * (-J)  # ferromagnetic
        elif J_disorder_type == "uniform":
            J_couplings = np.random.uniform(-J, J, len(pairs))
        elif J_disorder_type == "bimodal":
            J_couplings = np.random.choice([-J, J], len(pairs))
        elif J_disorder_type == "Gaussian":
            J_couplings = np.random.normal(0, J, len(pairs))
        if onsite_disorder_type == "uniform":
            site_disorder = np.random.uniform(-site_disord, site_disord, Ns)
        elif onsite_disorder_type == "bimodal":
            site_disorder = np.random.choice([-site_disord, site_disord], Ns)
        elif onsite_disorder_type == "Gaussian":
            site_disorder = np.random.normal(0, site_disord, Ns)

        self.J_terms = [[J_couplings[i], *pairs[i]] for i in range(len(pairs))]
        # why do I need to flip the sign of one of these to get good fidelity?????
        self.hx_terms = [[hx, i] for i in range(Ns)]
        self.double_hx_terms = [[2 * hx, i] for i in range(Ns)]
        self.disorder_terms = [[site_disorder[i], i] for i in range(Ns)]

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
        return None

    def build_H1(self):
        """Build QuSpin Hamiltonian for H1, which is component being
        "turned on" in annealing problem"""
        return None

    def build_bare_H(self):
        """Method specific to this spin model to calculate
        the bare Hamiltonian (no controls or AGP)
        """
        s = [["z", self.disorder_terms], ["zz", self.J_terms]]
        d = [["x", self.hx_terms, sweep_sign, [self.schedule]]]
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_dlam_H(self):
        """Method for this particular spin model to calculate
        the $\lambda$ derivative of the Hamiltonian
        """
        s = [["x", self.double_hx_terms]]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.basis, **self.checks)

    def build_target_H(self):
        """Method for this particular spin model to return the target
        Hamiltonian after annealing is complete, in the
        most symmetric possible basis to get the ground state easier
        """
        s = [["z", self.disorder_terms], ["zz", self.J_terms], ["x", self.hx_terms]]
        d = []
        return quspin.operators.hamiltonian(s, d, basis=self.targ_basis, **self.checks)
