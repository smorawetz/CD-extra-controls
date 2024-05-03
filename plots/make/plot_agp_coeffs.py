import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy

from matplotlib.ticker import MaxNLocator

sys.path.append(os.environ["CD_CODE_DIR"])

from plots.plot_utils import std_settings
from tools.schedules import LinearSchedule, SmoothSchedule
from tools.symmetries import get_symm_op
from utils.file_naming import make_coeffs_fname, make_base_fname

std_settings()


############# plotting #############
def plot_agp_coeffs_v_grid_size(
    grid_sizes,
    Ns,
    model_name,
    H_params,
    symmetries,
    ctrls,
    agp_order,
    AGPtype,
    norm_type,
    sched,
    append_str,
):
    fig, ax = plt.subplots(figsize=(9, 6))
    for i in range(len(grid_sizes)):
        grid_size = grid_sizes[i]
        fname = (
            make_base_fname(
                Ns,
                model_name,
                H_params,
                symmetries,
                ctrls,
                agp_order,
                AGPtype,
                norm_type,
                grid_size,
                sched,
                append_str,
            )
            + "_coeffs"
        )
        if AGPtype == "commutator":
            tgrid = np.loadtxt("coeffs_data/{0}_alphas_tgrid.txt".format(fname))
            coeffs_grid = np.loadtxt(
                "coeffs_data/{0}_alphas_grid.txt".format(fname), ndmin=2
            )
        elif AGPtype == "krylov":
            tgrid = np.loadtxt("coeffs_data/{0}_lanc_coeffs_tgrid.txt".format(fname))
            coeffs_grid = np.loadtxt(
                "coeffs_data/{0}_gammas_grid.txt".format(fname), ndmin=2
            )
        else:
            raise ValueError(f"AGPtype {AGPtype} not recognized")

        lam_grid = sched.get_lam(tgrid)
        ax.plot(lam_grid, coeffs_grid, label=f"{grid_size} pts", linewidth=2)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\alpha$" if AGPtype == "commutator" else r"$\gamma$")
    fig.legend(frameon=False)
    fig.savefig(
        "plots/images/agp_coeffs_vs_grid_{0}_ord{1}.pdf".format(AGPtype, agp_order)
    )


############# params #############
Ns = 8
model_name = "LR_Ising_1D"
H_params = [1, 1, 2]

symms = ["translation_1d", "spin_inversion"]
symms_args = [[Ns], [Ns]]
symm_nums = [0, 0]
symmetries = {
    symms[i]: (get_symm_op(symms[i], *symms_args[i]), symm_nums[i])
    for i in range(len(symms))
}

ctrls = ["Hc1", "Hc2"]
ctrls_couplings = ["sin", "sin"]

AGPtype = "krylov"
norm_type = "trace"

# grid_size = 1000
coeffs_sched = SmoothSchedule(1)  # always use tau = 1 for grid save
coeffs_append_str = "optim_ctrls"

grid_sizes = [50, 100, 200, 300, 500, 1000, 2500]

plot_agp_coeffs_v_grid_size(
    grid_sizes,
    Ns,
    model_name,
    H_params,
    symmetries,
    ctrls,
    1,
    AGPtype,
    norm_type,
    coeffs_sched,
    coeffs_append_str,
)
