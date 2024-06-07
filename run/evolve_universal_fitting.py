import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import numpy as np
import quspin

from tools.schedules import LinearSchedule, SmoothSchedule
from tools.symmetries import get_symm_op
from utils.file_naming import make_base_fname

from tools.calc_universal_fit_coeffs import fit_universal_coeffs
from scripts.time_evolution_universal_coeffs import run_time_evolution_universal
from utils.file_naming import make_fit_coeffs_fname

###############################################################
## this is a generic example of how to run a pre made script ##
###############################################################

g = 0.00

# define the various parameters of the model/task
Ns = 10
model_name = "Field_Sensing_1D"
H_params = [1, 1, g]
boundary_conds = "periodic"

symms = ["translation_1d"]
symms_args = [[Ns]]
symm_nums = [0]
symmetries = {
    symms[i]: (get_symm_op(symms[i], *symms_args[i]), symm_nums[i])
    for i in range(len(symms))
}
target_symms = ["translation_1d", "spin_inversion"]
target_symms_args = [[Ns], [Ns]]
target_symm_nums = [0, 0]
target_symmetries = {
    target_symms[i]: (
        get_symm_op(target_symms[i], *target_symms_args[i]),
        target_symm_nums[i],
    )
    for i in range(len(target_symms))
}

model_kwargs = {}

# schedule will be for coeffs grid, or evolution depending on script
evolve_tau = 0.001
coeffs_tau = 1
evolve_sched = SmoothSchedule(evolve_tau)
coeffs_sched = SmoothSchedule(coeffs_tau)

ctrls = []
ctrls_couplings = []
ctrls_args = []

agp_order = 9
AGPtype = "commutator"
norm_type = "trace"
window_start = 0.5
window_end = 4.0

alphas = fit_universal_coeffs(agp_order, window_start, window_end)
coeffs_fname = make_fit_coeffs_fname(agp_order, window_start, window_end)

np.savetxt(
    "{0}/coeffs_data/{1}.txt".format(os.environ["CD_CODE_DIR"], coeffs_fname), alphas
)

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

# now load coefficients for the universal fitting

wfs_append_str = "universal_g{0:.6f}".format(g)

kwargs = {
    "save_wf": False,
    "coeffs_fname": coeffs_fname,
    "wfs_save_append_str": wfs_append_str,
    "print_fid": True,
}

final_wf = run_time_evolution_universal(*args, **kwargs)
