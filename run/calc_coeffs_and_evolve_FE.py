import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import numpy as np
import quspin


from scripts.calc_floquet_coeffs_from_comms import get_floquet_coeffs_from_comm_coeffs
from scripts.merge_data import run_FE_agp_coeffs_merge
from scripts.time_evolution_FE import run_time_evolution_FE

from tools.schedules import LinearSchedule, SmoothSchedule
from tools.symmetries import get_symm_op
from tools.calc_universal_fit_coeffs import fit_universal_coeffs
from utils.file_naming import (
    make_fit_coeffs_fname,
    make_file_name,
    make_FE_protocol_name,
    make_controls_name,
)

###############################################################
## this is a generic example of how to run a pre made script ##
###############################################################

# Ns = [6]
# coeffs_model_name = "XY_1D"
# coeffs_model_name = "TFIM_1D"
# coeffs_H_params = [1, 1]
Nx = 8
Ny = 2
Ns = [Nx, Ny]
coeffs_model_name = "XY_2D"
coeffs_H_params = [1, 1]
coeffs_boundary_conds = "periodic"

symms = ["dbl_translation_x_2d"]
# symms = ["translation_1d", "spin_inversion"]
# symms = []
symms_args = [[Ns], [Ns]]
symm_nums = [0, 0]
coeffs_symmetries = {
    symms[i]: (get_symm_op(symms[i], *symms_args[i]), symm_nums[i])
    for i in range(len(symms))
}
coeffs_symmetries["m"] = 0.0
coeffs_target_symmetries = coeffs_symmetries

# evolve_model_name = "XY_1D"
# evolve_model_name = "TFIM_1D"
evolve_model_name = "XY_2D"
evolve_H_params = [1, 1]
evolve_boundary_conds = coeffs_boundary_conds
evolve_symmetries = coeffs_symmetries
evolve_target_symmetries = coeffs_target_symmetries

model_kwargs = {}

# schedule will be for coeffs grid, or evolution depending on script
evolve_tau = 1
coeffs_tau = 1
evolve_sched = SmoothSchedule(evolve_tau)
coeffs_sched = SmoothSchedule(coeffs_tau)

ctrls = []
ctrls_couplings = []
ctrls_args = []

agp_order = 1
AGPtype = "floquet"
norm_type = "trace"

grid_size = 1000

# have generic list of args that get used for every function
args = (
    ## H params
    Ns,
    coeffs_model_name,
    coeffs_H_params,
    coeffs_boundary_conds,
    coeffs_symmetries,
    coeffs_target_symmetries,
    model_kwargs,
    ## schedule params
    coeffs_tau,
    coeffs_sched,
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

# omega0 = 2 * np.pi * 10.0
# omega = 2 * np.pi * 250.0

omega0 = 3.0
omega = 50.0

kwargs = {
    "omega0": omega0,
}

# compute coefficients and then merge them
tgrid, betas_grid = get_floquet_coeffs_from_comm_coeffs(*args, **kwargs)

run_FE_agp_coeffs_merge(
    Ns,
    coeffs_model_name,
    coeffs_H_params,
    coeffs_boundary_conds,
    coeffs_symmetries,
    coeffs_sched,
    ctrls,
    ctrls_couplings,
    ctrls_args,
    agp_order,
    AGPtype,
    norm_type,
    grid_size,
    omega0=omega0,
)

# now run evolution

args = (
    ## H params
    Ns,
    evolve_model_name,
    evolve_H_params,
    evolve_boundary_conds,
    evolve_symmetries,
    evolve_target_symmetries,
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


save_protocol_wf = False

coeffs_file_name = make_file_name(
    Ns,
    coeffs_model_name,
    coeffs_H_params,
    coeffs_symmetries,
    ctrls,
    coeffs_boundary_conds,
)
coeffs_protocol_name = make_FE_protocol_name(
    agp_order,
    0.0,  # this omega does not change coefficients so just pass None
    omega0,
    grid_size,
    coeffs_sched,
)
coeffs_controls_name = make_controls_name(ctrls_couplings, ctrls_args)

print_fid = True
print_states = False

kwargs = {
    "omega": omega,
    "omega0": omega0,
    "save_protocol_wf": save_protocol_wf,
    "coeffs_file_name": coeffs_file_name,
    "coeffs_protocol_name": coeffs_protocol_name,
    "coeffs_ctrls_name": coeffs_controls_name,
    "coeffs_sched": coeffs_sched,
    "print_fid": print_fid,
    "print_states": print_states,
}

final_wf, target_wf = run_time_evolution_FE(*args, **kwargs)
