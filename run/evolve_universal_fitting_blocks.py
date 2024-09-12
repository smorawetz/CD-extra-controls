import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import numpy as np
import quspin

from tools.schedules import LinearSchedule, SmoothSchedule
from tools.symmetries import get_symm_op
from tools.calc_universal_fit_coeffs import fit_universal_coeffs
from utils.file_naming import make_fit_coeffs_fname

from scripts.time_evolution_universal_blocks import run_time_evolution_universal_blocks

###############################################################
## this is a generic example of how to run a pre made script ##
###############################################################

# define the various parameters of the model/task
full_Ns = [100]
block_Ns = [2]
model_name = "TFIM_k_Block_Annealing_1D"
H_params = [1, 1, None]
boundary_conds = "periodic"
symmetries = {}
target_symmetries = symmetries
model_kwargs = {}

# schedule will be for coeffs grid, or evolution depending on script
evolve_tau = 0.001  # needs to be sufficiently fast
evolve_sched = SmoothSchedule(evolve_tau)

ctrls = []
ctrls_couplings = []
ctrls_args = []

# agp_order = 10
AGPtype = "chebyshev"
norm_type = "trace"
# window_start = 0.5
window_end = 4.0
rescale = 1 / window_end

listof_agp_orders = np.arange(1, 5 + 1, 1)
listof_deltas = np.logspace(-1.5, 0.5, 11)[:-2]

grid_size = 1000

# k blocks depends on model
kblocks = 2 * np.pi / full_Ns[0] * np.arange(0.5, full_Ns[0] // 2, 1)

for agp_order in np.arange(1, 5 + 1, 1):
    for window_start in np.logspace(-1.5, 0.5, 11)[:-2]:
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
            evolve_tau,
            evolve_sched,
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
            "rescale": rescale,
            "window_start": window_start,
            "window_end": window_end,
            "save_protocol_wf": False,
            "print_fid": False,
            "print_states": False,
        }

        final_wf_blocks, target_state_blocks = run_time_evolution_universal_blocks(
            *args, **kwargs
        )
