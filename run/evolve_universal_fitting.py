import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import numpy as np
import quspin

from tools.schedules import LinearSchedule, SmoothSchedule
from tools.symmetries import get_symm_op
from tools.calc_universal_fit_coeffs import fit_universal_coeffs
from utils.file_naming import make_fit_coeffs_fname

from scripts.time_evolution_universal_coeffs import run_time_evolution_universal

###############################################################
## this is a generic example of how to run a pre made script ##
###############################################################

# define the various parameters of the model/task
Ns = [2]
model_name = "TFIM_Disorder_1D"
H_params = [1, 1, 0.1, 0]  # seed 0 and disorder strength 0.1
boundary_conds = "periodic"

symms = []
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

agp_order = 3
AGPtype = "chebyshev"
norm_type = "trace"
window_start = 0.25
window_end = 1.0

rescale = 1 / window_end

coeffs = fit_universal_coeffs(agp_order, AGPtype, window_start, window_end)

coeffs_fname = make_fit_coeffs_fname(AGPtype, agp_order, window_start, window_end)
np.savetxt(
    "{0}/universal_coeffs/{1}.txt".format(os.environ["CD_CODE_DIR"], coeffs_fname),
    coeffs,
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

kwargs = {
    "rescale": rescale,
    "window_start": window_start,
    "window_end": window_end,
    "save_protocol_wf": False,
    "print_fid": True,
}

final_wf = run_time_evolution_universal(*args, **kwargs)
