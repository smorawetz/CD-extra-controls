import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import glob
import h5py
import numpy as np

from scripts.merge_data import run_universal_evolved_wfs_blocks_merge

from tools.schedules import SmoothSchedule
from tools.symmetries import get_symm_op


Ns = [1000]
model_name = "TFIM_k_Block_Annealing_1D"
H_params = [1, 1, None]
boundary_conds = "periodic"
symmetries = {}
target_symmetries = symmetries
model_kwargs = {}

## schedule params
tau = 0.001
sched = SmoothSchedule(tau)

## controls params
ctrls = []
ctrls_couplings = []
ctrls_args = []

## agp params
# agp_order = 3
AGPtype = "chebyshev"
norm_type = "trace"

grid_size = 1000

listof_agp_orders = np.arange(1, 100 + 1)

for agp_order in listof_agp_orders:
    opt_delta1s = np.loadtxt("TFIM_clean_opt_deltas.txt")
    delta2_slope, delta2_intercept = np.loadtxt("NNN_TFIM_noninteg_fit_params.txt")

    window_end = delta2_slope * agp_order + delta2_intercept
    window_start = window_end / 4.0 * opt_delta1s[agp_order - 1] * agp_order ** (-1)
    rescale = 1 / window_end

    run_universal_evolved_wfs_blocks_merge(
        Ns,
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
