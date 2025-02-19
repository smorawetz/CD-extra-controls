import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import glob
import h5py
import numpy as np

from scripts.merge_data import run_evolved_wfs_merge

from tools.schedules import LinearSchedule
from tools.symmetries import get_symm_op


model_name = "KZ_Large_S_Sensing_EndFM"
J = 1
h = 1
g = 1.0
H_params = [J, h, g]

symms = []
symms_args = []
symm_nums = []
symmetries = {
    symms[i]: (get_symm_op(symms[i], *symms_args[i]), symm_nums[i])
    for i in range(len(symms))
}
target_symmetries = symmetries
model_kwargs = {}

## schedule params
tau = 1000
sched = LinearSchedule(tau)

## controls params
ctrls = []
ctrls_couplings = []
ctrls_args = []

## agp params
agp_order = 0
AGPtype = None
norm_type = None
grid_size = 1000

listof_N = np.arange(2, 40 + 1, 2)
listof_Ns = [[N] for N in listof_N]

listof_taus = np.logspace(2, 4, 21)


for Ns in listof_Ns:
    for tau in listof_taus:
        sched = LinearSchedule(tau)
        run_evolved_wfs_merge(
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
