import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import numpy as np
import quspin

import time

from tools.schedules import LinearSchedule, SmoothSchedule
from tools.symmetries import get_symm_op
from tools.calc_universal_fit_coeffs import fit_universal_coeffs
from tools.lin_alg_calls import calc_fid
from utils.file_naming import make_fit_coeffs_fname

from scripts.time_evolution_noCD import run_time_evolution_noCD

###############################################################
## this is a generic example of how to run a pre made script ##
###############################################################

# define the various parameters of the model/task
Nx = 10
Ny = 2
Ns = [Nx, Ny]
model_name = "XY_2D"
H_params = [1, 1]
boundary_conds = "periodic"

symms = ["dbl_translation_x_2d", "dbl_translation_y_2d"]
symms_args = [[Ns], [Ns]]
symm_nums = [0, 0]
symmetries = {
    symms[i]: (get_symm_op(symms[i], *symms_args[i]), symm_nums[i])
    for i in range(len(symms))
}
symmetries["m"] = 0.0
target_symmetries = symmetries

model_kwargs = {}

# schedule will be for coeffs grid, or evolution depending on script
evolve_tau = 1
evolve_sched = LinearSchedule(evolve_tau)

ctrls = []
ctrls_couplings = []
ctrls_args = []

agp_order = 0
AGPtype = None
norm_type = None

grid_size = 1000

# have generic list of args that get used for every function
args = (
    ## H params
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    target_symmetries,
    model_kwargs,
    ## schedule params
    evolve_tau,
    evolve_sched,
    ## controls params
    ctrls,
    ctrls_couplings,
    ctrls_args,
    ## agp params
    agp_order,
    AGPtype,
    norm_type,
    ## simulation params
    grid_size,
)

# kwargs = {"save_protocol_wf": True, "print_fid": True, "print_states": False}
kwargs = {"save_protocol_wf": False, "print_fid": True, "print_states": False}

final_wf, target_wf = run_time_evolution_noCD(*args, **kwargs)
