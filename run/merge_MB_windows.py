import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import glob
import h5py
import numpy as np

from scripts.merge_data import run_opt_windows_merge

from tools.schedules import SmoothSchedule
from tools.symmetries import get_symm_op


Ns = [8]
model_name = "NNN_TFIM_1D"
H_params = [1, 0.25, 1]
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

model_kwargs = {}

# schedule will be for coeffs grid, or evolution depending on script
tau = 1.0
sched = SmoothSchedule(tau)

ctrls = []
ctrls_couplings = []
ctrls_args = []

Nlamvals = 101
lamvals = np.linspace(0, 1, Nlamvals)

for lamval in lamvals:
    run_opt_windows_merge(
        lamval,
        Ns,
        model_name,
        H_params,
        symmetries,
        sched,
        ctrls,
        ctrls_couplings,
        ctrls_args,
    )
