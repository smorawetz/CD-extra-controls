import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import numpy as np
import quspin


from scripts.calc_floquet_coeffs import get_floquet_coeffs
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

Ns = [10]
model_name = "NNN_TFIM_1D"
H_params = [1, 0.25, 1]  # seed 1 and disorder strength 0.1
boundary_conds = "periodic"

# Ns = [4]
# model_name = "TFIM_1D"
# H_params = [1, 1]  # seed 1 and disorder strength 0.1
# boundary_conds = "periodic"

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
evolve_tau = 0.1
coeffs_tau = 1
evolve_sched = SmoothSchedule(evolve_tau)
coeffs_sched = SmoothSchedule(coeffs_tau)

ctrls = []
ctrls_couplings = []
ctrls_args = []

agp_order = 2
AGPtype = "floquet"
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

mu = 1.0
omega0 = 2 * np.pi * 2.0
spec_fn_Ns = [10]

omega = 2 * np.pi * 100000.0

kwargs = {
    "mu": mu,
    "omega0": omega0,
    "spec_fn_Ns": spec_fn_Ns,
}

# compute coefficients and then merge them
tgrid, betas_grid = get_floquet_coeffs(*args, **kwargs)

run_FE_agp_coeffs_merge(
    Ns,
    model_name,
    H_params,
    symmetries,
    coeffs_sched,
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

# now run evolution

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


save_protocol_wf = False

coeffs_file_name = make_file_name(
    Ns, model_name, H_params, symmetries, ctrls, boundary_conds
)
coeffs_protocol_name = make_FE_protocol_name(
    agp_order,
    0.0,  # this omega does not change coefficients so just pass None
    mu,
    omega0,
    grid_size,
    coeffs_sched,
)
coeffs_controls_name = make_controls_name(ctrls_couplings, ctrls_args)

print_fid = True
print_states = True

kwargs = {
    "omega": omega,
    "mu": mu,
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
