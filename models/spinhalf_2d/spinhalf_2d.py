import os
import sys

import quspin

sys.path.append(os.environ["CD_CODE_DIR"])

from cd_hamiltonian import Hamiltonian_CD
from tools.connections import neighbours_2d


class SpinHalf_2D(Hamiltonian_CD):
    """Child class of Hamiltonian_CD for 1D spin-1/2 systems"""

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

        Ns = Nx * Ny
        self.basis = quspin.basis.spin_basis_general(Ns, S="1/2", **symmetries)
        self.targ_basis = quspin.basis.spin_basis_general(
            Ns, S="1/2", **target_symmetries
        )
        self.pairs = neighbours_2d(Nx, Ny, boundary_conds)

        self.model_type = "spin-1/2"
        self.model_dim = 2

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
