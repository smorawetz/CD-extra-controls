import os
import sys

import numpy as np
import scipy

sys.path.append(os.environ["CD_CODE_DIR"])

from agp.krylov_construction import op_norm
from tools.build_ham import build_ham
from tools.calc_coeffs import calc_lanc_coeffs_grid, calc_gammas_grid
from tools.schedules import LinearSchedule, SmoothSchedule
from tools.symmetries import get_symm_op
from utils.file_naming import make_coeffs_fname
from utils.grid_utils import get_coeffs_interp


def run_calc_gammas(
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    tau,
    sched,
    ctrls,
    ctrls_couplings,
    ctrls_args,
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
    ham.init_controls(ctrls, ctrls_couplings, ctrls_args)

    fname = make_coeffs_fname(
        ham, model_name, ctrls, AGPtype, norm_type, grid_size, sched, append_str
    )

    # now call function to compute alphas
    lanc_tgrid, lanc_grid = calc_lanc_coeffs_grid(
        ham,
        grid_size,
        sched,
        agp_order,
        norm_type,
        gs_func=ham.get_inst_gstate,
        save=True,
        fname=fname,
    )
    ham.lanc_interp = scipy.interpolate.interp1d(lanc_tgrid, lanc_grid, axis=0)
    tgrid, gammas_grid = calc_gammas_grid(
        ham,
        grid_size,
        sched,
        agp_order,
        norm_type,
        gs_func=ham.get_inst_gstate,
        save=True,
        fname=fname,
    )
    ham.gammas_interp = scipy.interpolate.interp1d(tgrid, gammas_grid, axis=0)

    import matplotlib.pyplot as plt

    # do a sanity check
    t_data = np.loadtxt("coeffs_data/{0}_gammas_tgrid.txt".format(fname))
    gammas_data = np.loadtxt("coeffs_data/{0}_gammas_grid.txt".format(fname))

    fig, ax = plt.subplots()
    if agp_order == 1:
        ax.plot(t_data, gammas_data[:], label="g1")
    elif agp_order == 2:
        ax.plot(t_data, gammas_data[:, 0], label="g1")
        ax.plot(t_data, gammas_data[:, 1], label="g2")
    # plt.savefig(f"gammas_{agp_order}agp_test.png")


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
# sched = SmoothSchedule(tau)

# ctrls = []
# ctrls_couplings = []
# ctrls_args = []

ctrls = ["Hc1", "Hc2"]
ctrls_couplings = ["sin", "sin"]
harmonics = [1, 1]
# 1st order optimal
# ctrls_coeffs = [-1.03440, 2.99858]
# 2nd order optimal
# ctrls_coeffs = [-1.14467, 1.60439]
# 3rd order optimal
ctrls_coeffs = [-1.99224, 2.51090]
ctrls_args = [
    [sched, harmonics[0], ctrls_coeffs[0]],
    [sched, harmonics[1], ctrls_coeffs[1]],
]

agp_order = 7
AGPtype = "krylov"
# AGPtype = "commutator"
norm_type = "trace"
# norm_type = "ground_state"

grid_size = 1000

# 50, 100, 200, 300, 500, 1000, 2500, 10000

append_str = "optim_ctrls"
# append_str = "no_ctrls"

run_calc_gammas(
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    tau,
    sched,
    ctrls,
    ctrls_couplings,
    ctrls_args,
    agp_order,
    AGPtype,
    norm_type,
    grid_size,
    append_str,
)
