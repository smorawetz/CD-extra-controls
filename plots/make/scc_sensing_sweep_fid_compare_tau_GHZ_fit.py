import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import quspin
import scipy

from matplotlib.ticker import MaxNLocator

sys.path.append(os.environ["CD_CODE_DIR"])

from plots.plot_utils import std_settings
from tools.build_ham import build_ham
from tools.lin_alg_calls import calc_fid
from tools.schedules import SmoothSchedule
from tools.symmetries import get_symm_op
from run.get_target_gstate import get_target_gstate
from utils.file_naming import make_base_fname, make_evolved_wfs_fname

std_settings()


############# plotting #############
def plot_sweep_fid_vs_g_compare_tau(
    listof_taus,
    listof_g,
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    target_symmetries,
    model_kwargs,
    ctrls,
    ctrls_couplings,
    ctrls_args,
    agp_order,
    AGPtype,
    norm_type,
    grid_size,
    coeffs_sched,
    evolve_sched,  # will be overwritten
    coeffs_fname,
    append_str,
):
    H_params[2] = 0
    ham = build_ham(
        model_name,
        Ns,
        H_params,
        boundary_conds,
        model_kwargs,
        agp_order,
        norm_type,
        evolve_sched,
        symmetries=symmetries,
        target_symmetries=target_symmetries,
    )
    targ_wf = ham.get_targ_gstate()

    fig, ax = plt.subplots(figsize=(9, 5))
    for tau in listof_taus:
        evolve_sched = SmoothSchedule(tau)
        gs = []
        fids = []
        for g in listof_g:
            H_params[2] = g
            target_symmetries = symmetries if g != 0 else target_symmetries
            ham = build_ham(
                model_name,
                Ns,
                H_params,
                boundary_conds,
                model_kwargs,
                agp_order,
                norm_type,
                evolve_sched,
                symmetries=symmetries,
                target_symmetries=target_symmetries,
            )

            wfs_fname = make_evolved_wfs_fname(
                ham,
                model_name,
                ctrls,
                AGPtype,
                norm_type,
                grid_size,
                evolve_sched.tau,
                append_str,
            )

            # uncomment below to compare with true GS with knowledge of g
            # targ_wf = ham.get_targ_gstate()

            final_wf_fname = "{0}/wfs_evolved_data/{1}.txt".format(
                os.environ["CD_CODE_DIR"], wfs_fname
            )
            if not os.path.isfile(final_wf_fname):
                continue
            final_wf = np.loadtxt(final_wf_fname, dtype=np.complex128)

            fid = calc_fid(final_wf, targ_wf)
            fids.append(fid)
            gs.append(g)
        ax.plot(gs, fids, label=r"$\tau = {0:.4f}$".format(tau), linewidth=2)

    ax.set_xlim(-1.05 * max(listof_g), 1.05 * max(listof_g))
    ax.set_xlabel(r"$g$")
    ax.set_ylabel(r"$\mathcal{F}$")
    fig.legend(frameon=False, ncol=2)
    plt.savefig(
        "plots/images/sweep_fid_compare_tau_GHZ_fit_N{0}_ord{1}.pdf".format(
            Ns, agp_order, evolve_sched.tau
        )
    )


############# params #############
min_logtau = -3
max_logtau = 3
num_taus = 61
listof_taus = np.logspace(min_logtau, max_logtau, num_taus)[0::20]
max_g = 1.0
num_gs = 201
listof_gs = np.linspace(-max_g, max_g, num_gs)

# define the various parameters of the model/task
Ns = 8
model_name = "Field_Sensing_1D_Sweep"
H_params = [1, 1, 0.0, 0.0, 0]
boundary_conds = "periodic"

symms = ["translation_1d"]
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

agp_order = Ns // 2
AGPtype = "krylov"
norm_type = "trace"

grid_size = 1000
append_str = "GHZ_opt"

coeffs_model_name = "Field_Sensing_1D_Sweep"
coeffs_H_params = [1, 1, 0.0, 0.0, 0]
symms = ["translation_1d", "spin_inversion"]
symms_args = [[Ns], [Ns]]
symm_nums = [0, 0]
coeffs_symmetries = {
    symms[i]: (get_symm_op(symms[i], *symms_args[i]), symm_nums[i])
    for i in range(len(symms))
}
coeffs_target_symmetries = coeffs_symmetries

coeffs_fname = (
    make_base_fname(
        Ns,
        coeffs_model_name,
        coeffs_H_params,
        coeffs_symmetries,
        ctrls,
        agp_order,
        AGPtype,
        norm_type,
        grid_size,
        coeffs_sched,
        append_str,
    )
    + "_coeffs"
)

plot_sweep_fid_vs_g_compare_tau(
    listof_taus,
    listof_gs,
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    target_symmetries,
    model_kwargs,
    ctrls,
    ctrls_couplings,
    ctrls_args,
    agp_order,
    AGPtype,
    norm_type,
    grid_size,
    coeffs_sched,
    evolve_sched,
    coeffs_fname,
    append_str,
)
