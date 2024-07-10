import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy

from matplotlib.ticker import MaxNLocator

sys.path.append(os.environ["CD_CODE_DIR"])

from plots.plot_utils import std_settings
from ham_controls.build_controls_ham import get_H_controls_gs
from tools.build_ham import build_ham
from tools.lin_alg_calls import calc_fid
from tools.schedules import LinearSchedule, SmoothSchedule
from tools.symmetries import get_symm_op
from utils.file_naming import make_evolved_wfs_fname

std_settings()


############# plotting #############
def plot_inst_gs_overlap_compare_order(
    listof_agp_orders,
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    model_kwargs,
    ctrls,
    ctrls_couplings,
    ctrls_args,
    AGPtype,
    norm_type,
    sched,
    append_str,
    grid_size,
    rescale,
):
    fig, ax = plt.subplots(figsize=(9, 6))
    for i in range(len(listof_agp_orders)):  # loop to compare grid sizes
        agp_order = listof_agp_orders[i]
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
            target_symmetries=symmetries,
            rescale=rescale,
        )
        interp_tgrid = np.linspace(0, sched.tau, grid_size + 1)
        interp_wfgrid = np.zeros((ham.basis.Ns, grid_size + 1), dtype=np.complex128)
        for i in range(grid_size + 1):
            interp_wfgrid[:, i] = get_H_controls_gs(
                ham, interp_tgrid[i], ctrls, ctrls_couplings, ctrls_args
            )
        wf_interp = scipy.interpolate.interp1d(interp_tgrid, interp_wfgrid, axis=1)

        tgrid = np.linspace(0, sched.tau, grid_size + 1)
        wfs_fname = make_evolved_wfs_fname(
            ham,
            model_name,
            ctrls,
            AGPtype,
            norm_type,
            grid_size,
            sched.tau,
            append_str,
        )
        dirname = "{0}/wfs_evolved_data/{1}".format(
            os.environ["CD_CODE_DIR"], wfs_fname
        )
        overlaps = []
        for t in tgrid:
            path_name = "{0}/t{1:.6f}.txt".format(dirname, t)
            evolved_wf = np.loadtxt(path_name, dtype=np.complex128)
            inst_gs = wf_interp(t)
            overlaps.append(calc_fid(evolved_wf, inst_gs))
        lamgrid = sched.get_lam(tgrid)
        ax.plot(
            lamgrid[:-101], overlaps[:-101], label=f"Order {agp_order}", linewidth=2
        )

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\langle \psi_0 \vert \psi(t) \rangle$")
    fig.legend(frameon=False, loc=[0.15, 0.2])
    fig.savefig(
        "plots/images/inst_gs_overlap_vs_order_{0}_N{1}.pdf".format(AGPtype, Ns)
    )


############# params #############
g = 0.1
Ns = 8
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

targ_symms = ["translation_1d", "spin_inversion"]
targ_symms_args = [[Ns], [Ns]]
targ_symm_nums = [0, 0]
target_symmetries = {
    targ_symms[i]: (get_symm_op(targ_symms[i], *targ_symms_args[i]), targ_symm_nums[i])
    for i in range(len(targ_symms))
}
target_symmetries = symmetries

model_kwargs = {"disorder_strength": 0, "disorder_seed": 0}

ctrls = []
ctrls_couplings = []
ctrls_args = []

AGPtype = "chebyshev"
window_start = 0.1
window_end = 16.0
rescale = 1 / window_end
norm_type = "trace"

grid_size = 1000

tau = 0.001
sched = SmoothSchedule(tau)  # always use tau = 1 for grid save

append_str = "universal_g{0:.6f}".format(g)

listof_agp_orders = [5, 10, 15, 20, 25, 30]


plot_inst_gs_overlap_compare_order(
    listof_agp_orders,
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    model_kwargs,
    ctrls,
    ctrls_couplings,
    ctrls_args,
    AGPtype,
    norm_type,
    sched,
    append_str,
    grid_size,
    rescale,
)
