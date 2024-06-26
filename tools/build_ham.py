import os
import sys

import pickle

sys.path.append(os.environ["CD_CODE_DIR"])

from cd_hamiltonian import Hamiltonian_CD

with open("{0}/dicts/models.pkl".format(os.environ["CD_CODE_DIR"]), "rb") as f:
    models_dict = pickle.load(f)


def build_ham(
    model_name,
    Ns,
    H_params,
    boundary_conds,
    model_kwargs,
    agp_order,
    norm_type,
    sched,
    symmetries=[],
    target_symmetries=[],
):
    """Build the CD_Hamiltonian object corresponding to the Hamiltonian
    which encodes the desired annealing protocol
    Parameters:
        model_name (str):           Name of the model encoding annealing protocol
        Ns (int):                   Number of sites in the system
        H_params (list):            List of Hamiltonian parameters
        boundary_conds (str):       Boundary conditions for the system
        model_kwargs (dict):        Dictionary of model-specific parameters
        agp_order (int):            Order of the AGP term
        norm_type (str):            Either "trace" or "ground_state" for the norm
        sched (Schedule):           Schedule for the protocol to follow
        symmetries (list):          List of symmetries of the full Hamiltonian
        target_symmetries (list):   List of symmetries of the target Hamiltonian
    """
    ham_class = models_dict[model_name]
    return ham_class(
        Ns,
        H_params,
        boundary_conds,
        agp_order,
        norm_type,
        sched,
        symmetries=symmetries,
        target_symmetries=target_symmetries,
        **model_kwargs
    )
