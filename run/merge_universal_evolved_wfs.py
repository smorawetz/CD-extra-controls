import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import glob
import h5py
import numpy as np

from scripts.merge_data import run_universal_evolved_wfs_merge

from tools.schedules import SmoothSchedule
from tools.symmetries import get_symm_op


Ns = [2]
model_name = "TFIM_Disorder_1D"
H_params = [1, 1, 0.1, 0]  # seed 0 and disorder strength 0.1
boundary_conds = "periodic"

symms = []
symms_args = [[Ns]]
symm_nums = [0]
symmetries = {
    symms[i]: (get_symm_op(symms[i], *symms_args[i]), symm_nums[i])
    for i in range(len(symms))
}
target_symms = ["translation_1d", "spin_inversion"]
target_symms_args = [[Ns], [Ns]]
target_symm_nums = [0, 0]
target_symmetries = {
    target_symms[i]: (
        get_symm_op(target_symms[i], *target_symms_args[i]),
        target_symm_nums[i],
    )
    for i in range(len(target_symms))
}

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
window_start = 0.25
window_end = 1.0
## simulation params
grid_size = 1000

run_universal_evolved_wfs_merge(
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
