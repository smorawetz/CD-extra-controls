import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import numpy as np
import quspin

from tools.schedules import LinearSchedule, SmoothSchedule
from tools.symmetries import get_symm_op
from tools.calc_universal_fit_coeffs import fit_universal_coeffs
from utils.file_naming import make_fit_coeffs_fname

from scripts.calc_commutator_coeffs_blocks import calc_comm_coeffs_blocks

###############################################################
## this is a generic example of how to run a pre made script ##
###############################################################

# define the various parameters of the model/task
full_Ns = [120]
Nc = 2
block_Ns = [2 * Nc]
seed = 0
model_name = "TFIM_k_Block_Cell_Random"
H_params = [1, 1, 0, full_Ns[0], None]

block_Ns = [2]
model_name = "TFIM_k_Block_Annealing_1D"
H_params = [1, 1, None]

boundary_conds = "periodic"
symmetries = {}
target_symmetries = symmetries
model_kwargs = {}

# schedule will be for coeffs grid, or evolution depending on script
coeffs_tau = 1.0  # needs to be sufficiently fast
coeffs_sched = SmoothSchedule(coeffs_tau)

ctrls = []
ctrls_couplings = []
ctrls_args = []

agp_order = 4
AGPtype = "commutator"
# AGPtype = "krylov"
norm_type = "trace"

grid_size = 1000

# k blocks depends on model
kblocks = 2 * np.pi / full_Ns[0] * np.arange(0.5, full_Ns[0] // (2 * Nc), 1)

# have generic list of args that get used for every function
args = (
    ## H params
    full_Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,  # no symmetries within blocks
    target_symmetries,
    model_kwargs,
    ## schedule params
    coeffs_tau,
    coeffs_sched,
    ## controls params
    ctrls,
    ctrls_couplings,
    ctrls_args,
    ## agp params
    agp_order,
    AGPtype,
    norm_type,
    ## simulation params
    grid_size,
)

# now load coefficients for the universal fitting
kwargs = {
    "kblocks": kblocks,
    "block_Ns": block_Ns,
}

final_wf_blocks, target_state_blocks = calc_comm_coeffs_blocks(*args, **kwargs)
