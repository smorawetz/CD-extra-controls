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
        Ns,
        H_params,
        boundary_conds,
        agp_order,
        schedule,
        symmetries={},
        target_symmetries={},
        agp_orthog=True,
        norm_type="trace",
    ):
        """
        Parameters:
            Ns (int):                   Number of spins in spin model
            H_params (listof float):    Parameters of the spin model,
                                        e.g. [1, 2] for J = 1 and h = 2
            boundary_cond (str):        Whether to use open ("open") or periodic
                                        ("periodic") boundary conditions. Defaults
                                        to open
            agp_order (int):            Order of AGP ansatz
            schedule (Schedule):        Schedule object that encodes $\lambda(t)$
            symmetries (dictof (np.array, int)):    Symmetries of the Hamiltonian, which
                                        include a symmetry operation on the lattice
                                        and an integer which labels the sector by the
                                        eigenvalue of the symmetry transformation
            target_symmetries (dictof (np.array, int)):     Same as above, but for the
                                        target ground state if it has different symmetry
            agp_orthog (bool):          Whether or not to construct the AGP in the Krylov
                                        basis or via the commutator expansion ansatz. If
                                        True, uses the Krylov basis. If False, uses the
                                        regular commutator expansion ansatz
            norm_type (str):            What type of norm to use in the AGP. "trace" gives
                                        infinite temperature AGP, "ground_state" gives zero
                                        temperature
        """
        super().__init__(
            Ns,
            H_params,
            boundary_conds,
            symmetries=symmetries,
            target_symmetries=target_symmetries,
        )
        self.schedule = schedule
        self.agp_order = agp_order
        self.agp_orthog = agp_orthog
        self.norm_type = norm_type
