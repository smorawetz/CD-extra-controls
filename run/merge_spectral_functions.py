import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import glob
import h5py
import numpy as np

from scripts.merge_data import run_spectral_functions_merge

from tools.schedules import SmoothSchedule
from tools.symmetries import get_symm_op


Ns = [12]
model_name = "XXZ_1D"
H_params = [1, 1]
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
symmetries["m"] = 0.0

model_kwargs = {}

# schedule will be for coeffs grid, or evolution depending on script
tau = 1.0
sched = SmoothSchedule(tau)

ctrls = []
ctrls_couplings = []
ctrls_args = []

lamval = 0.2

ground_state = True

run_spectral_functions_merge(
    lamval,
    Ns,
    model_name,
    H_params,
    symmetries,
    sched,
    ctrls,
    ctrls_couplings,
    ctrls_args,
    ground_state=ground_state,
)
