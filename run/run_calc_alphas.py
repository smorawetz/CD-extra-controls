import os
import sys

import numpy as np

sys.path.append(os.environ["CD_CODE_DIR"])

from tools.build_ham import build_ham
from tools.calc_coeffs import calc_alphas_grid
from tools.schedules import LinearSchedule
from tools.symmetries import get_symm_op
from utils.file_naming import make_coeffs_fname


def run_calc_alphas(
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symms,
    symms_args,
    symm_nums,
    tau,
    sched,
    ctrls,
    agp_order,
    AGPtype,
    norm_type,
    grid_size,
    append_str,
):
    ham = build_ham(
        model_name,
        Ns,
        H_params,
        boundary_conds,
        agp_order,
        norm_type,
        sched,
        symmetries,
        target_symmetries,
    )

    fname = make_coeffs_fname(
        ham, model_name, ctrls, AGPtype, norm_type, grid_size, sched, append_str
    )

    # now call function to compute alphas
    calc_alphas_grid(
        ham,
        fname,
        grid_size,
        sched,
        agp_order,
        norm_type,
        gs_func=ham.get_inst_gstate,
        save=True,
    )


# things to run here
Ns = 8
# model_name = "TFIM_1D"
model_name = "LR_Ising_1D"
# H_params = [1, 1]
H_params = [1, 1, 2]
boundary_conds = "periodic"

symms = ["translation_1d", "spin_inversion"]
symms_args = [[Ns], [Ns]]
symm_nums = [0, 0]
symmetries = {
    symms[i]: (get_symm_op(symms[i], *symms_args[i]), symm_nums[i])
    for i in range(len(symms))
}
target_symmetries = symmetries

tau = 1
sched = LinearSchedule(tau)
ctrls = []

agp_order = 8
AGPtype = "commutator"
norm_type = "trace"
# norm_type = "ground_state"

grid_size = 1000
append_str = "normal"

run_calc_alphas(
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symms,
    symms_args,
    symm_nums,
    tau,
    sched,
    ctrls,
    agp_order,
    AGPtype,
    norm_type,
    grid_size,
    append_str,
)
