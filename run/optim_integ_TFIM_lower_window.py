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

from scripts.time_evolution_universal_blocks import run_time_evolution_universal_blocks

###############################################################
## this is a generic example of how to run a pre made script ##
###############################################################

J = 1
h = 1
# define the various parameters of the model/task
full_Ns = [100]
block_Ns = [2]
model_name = "TFIM_k_Block_Annealing_1D"
H_params = [J, h, None]
boundary_conds = "periodic"
symmetries = {}
target_symmetries = symmetries
model_kwargs = {}

# schedule will be for coeffs grid, or evolution depending on script
evolve_tau = 0.001  # needs to be sufficiently fast
evolve_sched = SmoothSchedule(evolve_tau)

ctrls = []
ctrls_couplings = []
ctrls_args = []

agp_order = 48
AGPtype = "chebyshev"
norm_type = "trace"
# window_start = 0.5
window_end = 4.0
rescale = 1 / window_end

grid_size = 1000

# k blocks depends on model
kblocks = 2 * np.pi / full_Ns[0] * np.arange(0.5, full_Ns[0] // 2, 1)

# have generic list of args that get used for every function
args = (
    ## H params
    full_Ns,
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
# NOTE: want to optimize the window_start so fill w/ None to start
kwargs = {
    "kblocks": kblocks,
    "block_Ns": block_Ns,
    "rescale": rescale,
    "window_start": None,
    "window_end": window_end,
    "save_protocol_wf": False,
    "print_fid": False,
    "print_states": False,
}

# define dirname
save_dirname = "{0}/data_dump".format(os.environ["DATA_DIR"])
# define file and controls name to be used in saving data
file_name = make_file_name(full_Ns, model_name, H_params, symmetries, ctrls)
protocol_name = make_fitting_protocol_name(AGPtype, agp_order, evolve_sched)
controls_name = make_controls_name(ctrls_couplings, ctrls_args)
# define things needed to save data
full_info_fname = combine_names(file_name, protocol_name, controls_name)
optim_list_fname = "{0}/window_optim/{1}.txt".format(save_dirname, full_info_fname)


def optim_func(log_window_start, args, kwargs):
    window_start = np.exp(log_window_start)
    kwargs["window_start"] = window_start
    final_wf_blocks, target_wf_blocks = run_time_evolution_universal_blocks(
        *args, **kwargs
    )
    fid = 1
    for n in range(len(final_wf_blocks[:, 0])):
        fid *= calc_fid(final_wf_blocks[n, 1:], target_wf_blocks[n, 1:])

    # add a line to file with fid data
    window_arr = np.array([window_start, window_end])
    with open(optim_list_fname, "a") as data_file:
        np.savetxt(data_file, np.expand_dims(np.array([*window_arr, fid]), axis=0))
    print("delta1: ", window_start, "delta2: ", window_end, "fid: ", fid)
    # minimize -log(fid) since easier with small numbers and log is monotonic
    return -np.log(fid)


scipy.optimize.minimize_scalar(
    optim_func,
    args=(args, kwargs),
    bracket=(0, -1),
    method="golden",
    options={"disp": True},
)
