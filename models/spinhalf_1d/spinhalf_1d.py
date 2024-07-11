import os
import sys

import quspin

sys.path.append(os.environ["CD_CODE_DIR"])

from cd_hamiltonian import Hamiltonian_CD
from tools.connections import neighbours_1d, triplets_1d


class SpinHalf_1D(Hamiltonian_CD):
    """Child class of Hamiltonian_CD for 1D spin-1/2 systems"""

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

        self.basis = quspin.basis.spin_basis_general(Ns, S="1/2", **symmetries)
        self.targ_basis = quspin.basis.spin_basis_general(
            Ns, S="1/2", **target_symmetries
        )
        self.pairs = neighbours_1d(Ns, boundary_conds)
        self.triplets = triplets_1d(Ns, boundary_conds)

        self.model_type = "spin-1/2"
        self.model_dim = 1

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
