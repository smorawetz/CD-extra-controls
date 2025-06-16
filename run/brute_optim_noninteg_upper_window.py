import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import numpy as np
import quspin
import scipy

from tools.build_ham import build_ham
from tools.calc_universal_fit_coeffs import fit_universal_coeffs
from tools.lin_alg_calls import calc_fid
from tools.schedules import SmoothSchedule
from tools.symmetries import get_symm_op
from utils.file_naming import (
    make_file_name,
    make_fitting_protocol_name,
    make_controls_name,
    combine_names,
)

from scripts.time_evolution_universal import run_time_evolution_universal

###############################################################
## this is a generic example of how to run a pre made script ##
###############################################################

J = 1
Delta = 1
# define the various parameters of the model/task
Ns = [16]
model_name = "XXZ_1D"
H_params = [J, Delta]
boundary_conds = "periodic"
symms = ["translation_1d", "spin_inversion"]
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
evolve_tau = 0.001  # needs to be sufficiently fast
evolve_sched = SmoothSchedule(evolve_tau)

ctrls = []
ctrls_couplings = []
ctrls_args = []

opt_deltas = np.loadtxt("TFIM_clean_opt_deltas.txt")

agp_order = 1
AGPtype = "chebyshev"
norm_type = "trace"
base_window_start = opt_deltas[agp_order - 1]

# choosing 4.0 below means keeping the same ratio of delta2 / delta1 as TFIM
single_part_exc = 4.0
single_part_rescale = single_part_exc / 4.0

grid_size = 1000

# have generic list of args that get used for every function
args = (
    ## H params
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,  # no symmetries within blocks
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
# NOTE: want to optimize the windows so fill w/ None to start
kwargs = {
    "rescale": None,
    "window_start": None,
    "window_end": None,
    "save_protocol_wf": False,
    "print_fid": False,
}

# define dirname
save_dirname = "{0}/data_dump".format(os.environ["DATA_DIR"])
# define file and controls name to be used in saving data
file_name = make_file_name(Ns, model_name, H_params, symmetries, ctrls, boundary_conds)
protocol_name = make_fitting_protocol_name(AGPtype, agp_order, evolve_sched)
controls_name = make_controls_name(ctrls_couplings, ctrls_args)
# define things needed to save data
full_info_fname = combine_names(file_name, protocol_name, controls_name)
optim_list_fname = "{0}/brute_window_optim/{1}.txt".format(
    save_dirname, full_info_fname
)

init_window_end = 3.0
final_window_end = 7.0
window_end_step = 0.01
listof_window_ends = np.arange(
    init_window_end, final_window_end + window_end_step, window_end_step
)

# pick one of them here
window_end = listof_window_ends[0]

window_start = base_window_start * window_end * single_part_rescale / 4.0
kwargs["rescale"] = 1 / window_end
kwargs["window_start"] = window_start
kwargs["window_end"] = window_end
final_wf, target_wf = run_time_evolution_universal(*args, **kwargs)
fid = calc_fid(final_wf, target_wf)

# add a line to file with fid data
window_arr = np.array([window_start, window_end])
with open(optim_list_fname, "a") as data_file:
    np.savetxt(data_file, np.expand_dims(np.array([*window_arr, fid]), axis=0))
print("delta1: ", window_start, "delta2: ", window_end, "fid: ", fid)
