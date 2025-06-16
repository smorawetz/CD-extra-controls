import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import numpy as np

from tools.schedules import LinearSchedule, SmoothSchedule
from tools.symmetries import get_symm_op
from utils.file_naming import make_file_name, make_protocol_name, make_controls_name

from scripts.time_evolution import run_time_evolution

# define the various parameters of the model/task
Ns = [4]
model_name = "TFIM_1D"
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
target_symmetries = symmetries

model_kwargs = {}

# schedule will be for coeffs grid, or evolution depending on script
evolve_tau = 0.01
sched = SmoothSchedule(evolve_tau)
coeffs_tau = 1.0
coeffs_sched = SmoothSchedule(coeffs_tau)

ctrls = []
ctrls_couplings = []
ctrls_args = []

agp_order = 2
AGPtype = "krylov"
norm_type = "trace"

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
    sched,
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

coeffs_file_name = make_file_name(
    Ns, model_name, H_params, symmetries, ctrls, boundary_conds
)
coeffs_protocol_name = make_protocol_name(
    AGPtype, norm_type, agp_order, grid_size, coeffs_sched
)
coeffs_ctrls_name = make_controls_name(ctrls_couplings, ctrls_args)

# TODO: need to expand functionality to use different coefficients to evolve
# now define the kwargs which are specific to each script

kwargs = {
    "save_protocol_wf": True,
    "coeffs_file_name": coeffs_file_name,
    "coeffs_protocol_name": coeffs_protocol_name,
    "coeffs_ctrls_name": coeffs_ctrls_name,
    "coeffs_sched": coeffs_sched,
    "print_fid": True,
}

final_state = run_time_evolution(*args, **kwargs)
