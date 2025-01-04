import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import numpy as np
import quspin

from tools.schedules import LinearSchedule, SmoothSchedule
from tools.symmetries import get_symm_op
from tools.calc_universal_fit_coeffs import fit_universal_coeffs
from utils.file_naming import make_file_name, make_protocol_name, make_controls_name

from scripts.time_evolution_blocks import run_time_evolution_blocks

###############################################################
## this is a generic example of how to run a pre made script ##
###############################################################

full_Ns = [120]
Nc = 2
block_Ns = [2 * Nc]
seed = 0
model_name = "TFIM_k_Block_Cell_Random"
H_params = [1, 1, 0, full_Ns[0], None]
boundary_conds = "periodic"
symmetries = {}
target_symmetries = symmetries
model_kwargs = {}

# schedule will be for coeffs grid, or evolution depending on script
evolve_tau = 0.001
sched = SmoothSchedule(evolve_tau)
coeffs_tau = 1.0
coeffs_sched = SmoothSchedule(coeffs_tau)

ctrls = []
ctrls_couplings = []
ctrls_args = []

agp_order = 3
AGPtype = "commutator"
# AGPtype = "krylov"
norm_type = "trace"

grid_size = 1000

# k blocks depends on model
kblocks = 2 * np.pi / full_Ns[0] * np.arange(0.5, full_Ns[0] // (2 * Nc), 1)

# have generic list of args that get used for every function
args = (
    ## H params
    block_Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    target_symmetries,
    model_kwargs,
    ## schedule params
    evolve_tau,
    sched,
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

H_params[-1] = "all"
coeffs_file_name = make_file_name(full_Ns, model_name, H_params, symmetries, ctrls)
coeffs_protocol_name = make_protocol_name(
    AGPtype, norm_type, agp_order, grid_size, coeffs_sched
)
coeffs_ctrls_name = make_controls_name(ctrls_couplings, ctrls_args)

kwargs = {
    "kblocks": kblocks,
    "block_Ns": block_Ns,
    "save_protocol_wf": False,
    "coeffs_file_name": coeffs_file_name,
    "coeffs_protocol_name": coeffs_protocol_name,
    "coeffs_ctrls_name": coeffs_ctrls_name,
    "coeffs_sched": coeffs_sched,
    "print_fid": True,
}

final_state = run_time_evolution_blocks(*args, **kwargs)
