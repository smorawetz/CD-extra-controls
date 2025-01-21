import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import glob
import h5py
import numpy as np

from scripts.merge_data import run_universal_evolved_wfs_merge

from tools.schedules import SmoothSchedule
from tools.symmetries import get_symm_op


# Ns = [4]
model_name = "NNN_TFIM_1D"
H_params = [1, 0.25, 1]  # seed 0 and disorder strength 0.1
boundary_conds = "periodic"

## schedule params
tau = 0.01
sched = SmoothSchedule(tau)

## controls params
ctrls = []
ctrls_couplings = []
ctrls_args = []

## agp params
# agp_order = 3
AGPtype = "chebyshev"
norm_type = "trace"

## simulation params
grid_size = 1000

listof_Ns = [[Ns] for Ns in np.arange(4, 16 + 1)]
listof_symmetries = []
listof_target_symmetries = []
for N in listof_Ns:
    Ns = N
    symms = ["translation_1d", "spin_inversion"]
    symms_args = [[Ns], [Ns]]
    symm_nums = [0, 0]
    symmetries = {
        symms[i]: (get_symm_op(symms[i], *symms_args[i]), symm_nums[i])
        for i in range(len(symms))
    }
    listof_symmetries.append(symmetries)
    listof_target_symmetries.append(symmetries)

listof_agp_orders = np.arange(6, 10 + 1)

listof_upper_windows = np.arange(11.0, 15.0 + 0.1, 0.1)

for window_end in listof_upper_windows:
    for agp_order in listof_agp_orders:
        opt_ords = np.loadtxt("TFIM_clean_opt_agp_orders.txt")
        opt_deltas = np.loadtxt("TFIM_clean_opt_deltas.txt")
        ind = np.where(opt_ords == agp_order)[0][0]
        window_start = opt_deltas[ind] * window_end / 4.0
        for j in range(len(listof_Ns)):
            Ns = listof_Ns[j]
            symmetries = listof_symmetries[j]
            target_symmetries = listof_target_symmetries[j]

            try:  # might not exist, in which case just merge other files
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
            except:
                continue
            else:
                print(
                    "N = {0}, agp_order = {1}, window_end = {2}".format(
                        Ns, agp_order, window_end
                    )
                )
