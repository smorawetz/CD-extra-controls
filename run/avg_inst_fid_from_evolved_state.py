import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import numpy as np

from tools.build_ham import build_ham
from tools.lin_alg_calls import calc_fid
from tools.symmetries import get_symm_op
from tools.schedules import SmoothSchedule

from utils.file_IO import load_data_evolved_wfs
from utils.file_naming import (
    make_file_name,
    make_universal_protocol_name,
    make_controls_name,
    combine_names,
)


# just pass to this the relevant file names, can be loaded from somewhere else
def avg_inst_log_fid(ham, file_name, protocol_name, controls_name):
    _, tgrid, full_wf = load_data_evolved_wfs(
        file_name, protocol_name, controls_name, get_full_wf=True
    )
    inst_wfs = np.zeros_like(full_wf)
    fids = []
    for i in range(len(tgrid)):
        targ_wf = ham.get_bare_inst_gstate(tgrid[i])
        evol_wf = full_wf[i, :]
        fids.append(calc_fid(targ_wf, evol_wf))
    save_dirname = "{0}/data_dump/avg_fids/".format(os.environ["DATA_DIR"])
    save_dirname = save_dirname + combine_names(file_name, protocol_name, controls_name)
    np.savetxt(save_dirname, np.array(fids))
    return None


def compute_universal_protocol_avg_fid(
    ## H params
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,  # no symmetries within blocks
    target_symmetries,
    model_kwargs,
    ## schedule params
    tau,
    sched,
    ## controls params
    ctrls,
    ctrls_couplings,
    ctrls_args,
    ## agp params
    agp_order,
    AGPtype,
    norm_type,
    window_start,
    window_end,
    ## simulation params
    grid_size,
):
    ham = build_ham(
        model_name,
        Ns,
        H_params,
        boundary_conds,
        model_kwargs,
        agp_order,
        norm_type,
        sched,
        symmetries=symmetries,
        target_symmetries=target_symmetries,
    )
    file_name = make_file_name(
        Ns, model_name, H_params, symmetries, ctrls, boundary_conds
    )
    protocol_name = make_universal_protocol_name(
        AGPtype, norm_type, agp_order, window_start, window_end, grid_size, sched
    )
    controls_name = make_controls_name(ctrls_couplings, ctrls_args)
    avg_inst_log_fid(ham, file_name, protocol_name, controls_name)
    return None


# define the various parameters of the model/task
Ns = [16]
model_name = "XXZ_1D"
J = 1
Delta = 1
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
tau = 0.001  # needs to be sufficiently fast
# evolve_tau = 1  # needs to be sufficiently fast
sched = SmoothSchedule(tau)

ctrls = []
ctrls_couplings = []
ctrls_args = []

opt_deltas = np.loadtxt("TFIM_clean_opt_deltas.txt")

agp_order = 1
AGPtype = "chebyshev"
norm_type = "trace"
base_window_start = opt_deltas[agp_order - 1]

# pick one of the possible window ends to compute the average fidelity for
init_window_end = 3.0
final_window_end = 7.0
window_end_step = 0.01
listof_window_ends = np.arange(
    init_window_end, final_window_end + window_end_step, window_end_step
)

window_end = listof_window_ends[0]
window_start = base_window_start * window_end / 4.0

grid_size = 1000


compute_universal_protocol_avg_fid(
    ## H params
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,  # no symmetries within blocks
    target_symmetries,
    model_kwargs,
    ## schedule params
    tau,
    sched,
    ## controls params
    ctrls,
    ctrls_couplings,
    ctrls_args,
    ## agp params
    agp_order,
    AGPtype,
    norm_type,
    window_start,
    window_end,
    ## simulation params
    grid_size,
)
