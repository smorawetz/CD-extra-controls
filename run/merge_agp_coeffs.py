import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import glob
import h5py
import numpy as np

from scripts.merge_data import run_agp_coeffs_merge

from tools.schedules import SmoothSchedule
from tools.symmetries import get_symm_op


Nx = 4
Ny = 2
Ns = [Nx, Ny]
model_name = "Disorder_Ising_2D"
H_params = [1, 1, 0.0, 0]  # seed 1 and disorder strength 0.1
symms = []  # disorder will in general break symmetry
symms_args = []
symm_nums = []
symmetries = {
    symms[i]: (get_symm_op(symms[i], *symms_args[i]), symm_nums[i])
    for i in range(len(symms))
}
target_symmetries = symmetries

## schedule params
tau = 1
sched = SmoothSchedule(tau)

## controls params
ctrls = []
ctrls_couplings = []
ctrls_args = []

## agp params
agp_order = 1
# AGPtype = "commutator"
AGPtype = "krylov"
norm_type = "trace"
## simulation params
grid_size = 1000

run_agp_coeffs_merge(
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
