import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import glob
import h5py
import numpy as np

from scripts.merge_data import run_universal_evolved_wfs_blocks_merge

from tools.schedules import SmoothSchedule
from tools.symmetries import get_symm_op


############# params #############
# define the various parameters of the model/task
full_Ns = [100]
model_name = "TFIM_k_Block_Annealing_1D"
H_params = [1, 1, None]
boundary_conds = "periodic"
symmetries = {}
model_kwargs = {}

ctrls = []
ctrls_couplings = []
ctrls_args = []

# agp_order = 10
AGPtype = "chebyshev"
norm_type = "trace"

# schedule will be for coeffs grid, or evolution depending on script
evolve_tau = 0.001  # needs to be sufficiently fast
coeffs_tau = 1
evolve_sched = SmoothSchedule(evolve_tau)
coeffs_sched = SmoothSchedule(coeffs_tau)

grid_size = 1000

## schedule params
tau = 0.001
sched = SmoothSchedule(tau)

## controls params
ctrls = []
ctrls_couplings = []
ctrls_args = []

## agp params
agp_order = 3
AGPtype = "chebyshev"
norm_type = "trace"
## simulation params
grid_size = 1000

window_end = 4.0
for window_start in np.logspace(-1.5, 0.5, 11)[:-2]:
    for agp_order in range(1, 5 + 1):
        run_universal_evolved_wfs_blocks_merge(
            full_Ns,
            model_name,
            H_params,
            symmetries,
            sched,
            ctrls,
            ctrls_couplings,
            ctrls_args,
            agp_order,
            AGPtype,
            norm_type,
            window_start,
            window_end,
            grid_size,
        )
