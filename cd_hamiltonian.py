import pickle
import os

import numpy as np
import scipy

import quspin

from base_hamiltonian import Base_Hamiltonian


class Hamiltonian_CD(Base_Hamiltonian):
    """This is a "Hamiltonian" child class of the Base_Hamiltonian class,
    with the additional ability to construct a counterdiabatic term"""

    def __init__(
        self,
        Nspins,
        H_params,
        boundary_conds,
        agp_order,
        agp_orthog=True,
        norm_type="trace",
    ):
        """
        Parameters:
            Nspins (int):               Number of spins in spin model
            H_params (listof float):    Parameters of the spin model,
                                        e.g. [1, 2] for J = 1 and h = 2
            boundary_cond (str):        Whether to use open ("open") or periodic
                                        ("periodic") boundary conditions. Defaults
                                        to open
            agp_order (int):            Order of AGP ansatz
            agp_orthog (bool):          Whether or not to construct the AGP in the Krylov
                                        basis or via the commutator expansion ansatz. If
                                        True, uses the Krylov basis. If False, uses the
                                        regular commutator expansion ansatz
            norm_type (str):            What type of norm to use in the AGP. "trace" gives
                                        infinite temperature AGP, "ground_state" gives zero
                                        temperature
        """
        super().__init__(Nspins, H_params, boundary_conds)
        self.agp_order = agp_order
        self.agp_orthog = agp_orthog
        self.norm_type = norm_type
