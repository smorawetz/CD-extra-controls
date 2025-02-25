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
from utils.file_naming import make_data_dump_name, combine_names
from utils.file_IO import load_data_evolved_wfs

std_settings()


############# plotting #############
def plot_inst_gs_overlap_compare_grid(
    grid_sizes,
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    model_kwargs,
    ctrls,
    ctrls_couplings,
    ctrls_args,
    agp_order,
    AGPtype,
    norm_type,
    sched,
    append_str,
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
        target_symmetries=symmetries,
    )
    interp_tgrid = np.linspace(0, sched.tau, max(grid_sizes))
    interp_wfgrid = np.zeros((ham.basis.Ns, max(grid_sizes)), dtype=np.complex128)
    for i in range(max(grid_sizes)):
        interp_wfgrid[:, i] = get_H_controls_gs(
            ham, interp_tgrid[i], ctrls, ctrls_couplings, ctrls_args
        )
    wf_interp = scipy.interpolate.interp1d(interp_tgrid, interp_wfgrid, axis=1)

    fig, ax = plt.subplots(figsize=(9, 6))
    for i in range(len(grid_sizes)):  # loop to compare grid sizes
        grid_size = grid_sizes[i]
        names_list = make_data_dump_name(
            Ns,
            model_name,
            H_params,
            symmetries,
            sched,
            ctrls,
            ctrls_couplings,
            ctrls_args,
            agp_order,
            AGPtype,
            norm_type,
            grid_size,
        )
        final_wf, tgrid, full_evolved_wf = load_data_evolved_wfs(
            *names_list, get_full_wf=True
        )
        overlaps = []
        for i in range(grid_size + 1):
            t = tgrid[i]
            evolved_wf = full_evolved_wf[i, :]
            inst_gs = wf_interp(t)
            overlaps.append(calc_fid(evolved_wf, inst_gs))
        lamgrid = sched.get_lam(tgrid)
        ax.plot(lamgrid, overlaps, label=f"{grid_size} pts", linewidth=2)

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\langle \psi_0 \vert \psi(t) \rangle$")
    fig.legend(frameon=False)
    fig.savefig(
        "plots/images/inst_gs_overlap_vs_grid_{0}_ord{1}.pdf".format(AGPtype, agp_order)
    )


############# params #############
Ns = [8]
model_name = "XXZ_1D"
H_params = [1, 1]
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

ctrls = []
ctrls_couplings = []
ctrls_args = []

tau = 0.01
sched = SmoothSchedule(tau)  # always use tau = 1 for grid save

append_str = "std"

agp_order = 1
AGPtype = "commutator"
norm_type = "trace"

# grid_sizes = [50, 100, 200, 300, 500, 1000, 2500]
# grid_sizes = [100, 500, 1000, 10000]
grid_sizes = [1000]

plot_inst_gs_overlap_compare_grid(
    grid_sizes,
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    model_kwargs,
    ctrls,
    ctrls_couplings,
    ctrls_args,
    agp_order,
    AGPtype,
    norm_type,
    sched,
    append_str,
)
