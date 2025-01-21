import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import glob
import h5py
import numpy as np

from scripts.merge_data import run_FE_agp_coeffs_merge

from tools.schedules import SmoothSchedule
from tools.symmetries import get_symm_op


# define the various parameters of the model/task
Ns = [10]
model_name = "NNN_TFIM_1D"
H_params = [1, 0.25, 1]  # seed 1 and disorder strength 0.1
boundary_conds = "periodic"

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
target_symmetries = symmetries

model_kwargs = {}

# schedule will be for coeffs grid, or evolution depending on script
tau = 1
sched = SmoothSchedule(tau)

ctrls = []
ctrls_couplings = []
ctrls_args = []

agp_order = 7
AGPtype = "floquet"
norm_type = "trace"

grid_size = 1000

mu = 1.0
omega0 = 1.0

run_FE_agp_coeffs_merge(
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
    mu=mu,
    omega0=omega0,
)
