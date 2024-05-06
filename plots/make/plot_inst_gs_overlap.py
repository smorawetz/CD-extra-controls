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
from utils.file_naming import make_base_fname

std_settings()


import time


############# plotting #############
def plot_inst_gs_overlap_compare_grid(
    grid_sizes,
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    ctrls,
    ctrls_couplings,
    ctrls_args,
    agp_order,
    AGPtype,
    norm_type,
    sched,
    append_str,
):
    time1 = time.time()
    ham = build_ham(
        model_name,
        Ns,
        H_params,
        boundary_conds,
        agp_order,
        norm_type,
        sched,
        symmetries=symmetries,
        target_symmetries=symmetries,
    )
    time2 = time.time()
    print("time to build ham is ", time2 - time1)
    interp_tgrid = np.linspace(0, sched.tau, max(grid_sizes) + 1)
    interp_wfgrid = np.zeros((ham.basis.Ns, max(grid_sizes) + 1), dtype=np.complex128)
    for i in range(max(grid_sizes) + 1):
        interp_wfgrid[:, i] = get_H_controls_gs(
            ham, interp_tgrid[i], ctrls, ctrls_couplings, ctrls_args
        )
    wf_interp = scipy.interpolate.interp1d(interp_tgrid, interp_wfgrid, axis=1)
    time3 = time.time()
    print("time to get inst gs interp grid is ", time3 - time2)

    fig, ax = plt.subplots(figsize=(9, 6))
    for i in range(len(grid_sizes)):  # loop to compare grid sizes
        grid_size = grid_sizes[i]
        tgrid = np.linspace(0, sched.tau, grid_size + 1)
        wf_fname = (
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
            # + "_evolved_wf"
            + "_evoled_wf"
        )
        dirname = "wfs_evolved_data/{0}".format(wf_fname)
        overlaps = []
        for t in tgrid:
            path_name = "{0}/t{1:.6f}.txt".format(dirname, t)
            evolved_wf = np.loadtxt(path_name, dtype=np.complex128)
            inst_gs = wf_interp(t)
            overlaps.append(calc_fid(evolved_wf, inst_gs))
        ax.plot(tgrid, overlaps, label=f"{grid_size} pts", linewidth=2)

    time4 = time.time()
    print("time to calculate overlaps is ", time4 - time3)
    # ax.set_xlabel(r"$\lambda$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\langle \psi_0 \vert \psi(t) \rangle$")
    fig.legend(frameon=False)
    fig.savefig(
        "plots/images/inst_gs_overlap_vs_grid_{0}_ord{1}.pdf".format(AGPtype, agp_order)
    )


############# params #############
Ns = 8
model_name = "LR_Ising_1D"
H_params = [1, 1, 2]
boundary_conds = "periodic"

symms = ["translation_1d", "spin_inversion"]
symms_args = [[Ns], [Ns]]
symm_nums = [0, 0]
symmetries = {
    symms[i]: (get_symm_op(symms[i], *symms_args[i]), symm_nums[i])
    for i in range(len(symms))
}

# ctrls = []
# ctrls_couplings = []
# ctrls_args = []

tau = 0.01
sched = SmoothSchedule(tau)  # always use tau = 1 for grid save
append_str = "optim_ctrls"
# append_str = "no_ctrls"

ctrls = ["Hc1", "Hc2"]
ctrls_couplings = ["sin", "sin"]
harmonics = [1, 1]
# 1st order optimal
ctrls_coeffs = [-1.03440, 2.99858]
# 2nd order optimal
# ctrls_coeffs = [-1.14467, 1.60439]
# 3rd order optimal
# ctrls_coeffs = [-1.99224, 2.51090]
ctrls_args = [
    [sched, harmonics[0], ctrls_coeffs[0]],
    [sched, harmonics[1], ctrls_coeffs[1]],
]

agp_order = 1
AGPtype = "krylov"
norm_type = "trace"

# grid_sizes = [50, 100, 200, 300, 500, 1000, 2500]
grid_sizes = [100, 500, 1000, 10000]

plot_inst_gs_overlap_compare_grid(
    grid_sizes,
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    ctrls,
    ctrls_couplings,
    ctrls_args,
    agp_order,
    AGPtype,
    norm_type,
    sched,
    append_str,
)
