import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import glob
import h5py
import numpy as np

from scripts.merge_optimization_fids import run_merge

from tools.schedules import SmoothSchedule
from tools.symmetries import get_symm_op


Ns = 4
model_name = "TFIM_1D"
H_params = [1, 1]
symms = ["translation_1d", "spin_inversion"]
symms_args = [[Ns], [Ns]]
symm_nums = [0, 0]
symmetries = {
    symms[i]: (
        get_symm_op(symms[i], *symms_args[i]),
        symm_nums[i],
    )
    for i in range(len(symms))
}

## schedule params
tau = 0.001
sched = SmoothSchedule(tau)

## controls params
ctrls = ["Hc1", "Hc2"]
ctrls_couplings = ["sin", "sin"]
ctrls_harmonics = [1]
ctrls_args = [[None, 1.0, None], [None, 1.0, None]]

## agp params
agp_order = 1
# AGPtype = "commutator"
AGPtype = "krylov"
norm_type = "trace"
## simulation params
grid_size = 1000

run_merge(
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
    grid_size,
)
