import os
import sys

import pickle

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.environ["CD_CODE_DIR"])

from plots.plot_utils import std_settings

from tools.build_ham import build_ham
from tools.lin_alg_calls import calc_fid
from tools.schedules import SmoothSchedule
from tools.symmetries import get_symm_op
from utils.file_IO import load_data_evolved_wfs
from utils.file_naming import make_data_dump_name, make_universal_protocol_name

with open("{0}/dicts/fit_funcs.pkl".format(os.environ["CD_CODE_DIR"]), "rb") as f:
    fit_funcs_dict = pickle.load(f)

std_settings()


def compare_fid_vs_ord(
    listof_agp_orders,
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    target_symmetries,
    model_kwargs,
    sched,
    ctrls,
    ctrls_couplings,
    ctrls_args,
    AGPtype,
    window_start,
    window_end,
    norm_type,
    grid_size,
):
    ham = build_ham(
        model_name,
        Ns,
        H_params,
        boundary_conds,
        model_kwargs,
        1,  # AGP order doesn't matter for targ wf
        norm_type,
        sched,
        symmetries=symmetries,
        target_symmetries=target_symmetries,
    )
    targ_wf = ham.get_targ_gstate()

    fids = []
    for agp_order in listof_agp_orders:
        file_name, _, controls_name = make_data_dump_name(
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
        protocol_name = make_universal_protocol_name(
            AGPtype, norm_type, agp_order, window_start, window_end, grid_size, sched
        )
        names_list = [file_name, protocol_name, controls_name]
        final_wf, _, _ = load_data_evolved_wfs(*names_list, get_full_wf=False)

        fid = calc_fid(final_wf, targ_wf)
        fids.append(fid)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(listof_agp_orders, [np.log10(fid) for fid in fids], "k-", linewidth=2)
    ax.set_xlabel(r"AGP order")
    ax.set_ylabel(r"$\log\mathcal{F}$")
    plt.savefig(
        "plots/images/universal_fid_vs_order_{0}_N{1}_tau{2}.pdf".format(
            model_name, Ns, sched.tau
        )
    )


Ns = [6]
model_name = "TFIM_Sweep_Disorder_1D"
H_params = [0.2, 2, 0.5, 0.05, 0]  # seed 0 and disorder strength 0.1
boundary_conds = "periodic"

symms = []
symms_args = [[Ns], [Ns]]
symm_nums = [0, 0]
symmetries = {
    symms[i]: (get_symm_op(symms[i], *symms_args[i]), symm_nums[i])
    for i in range(len(symms))
}
target_symms = []
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
evolve_tau = 100
sched = SmoothSchedule(evolve_tau)

ctrls = []
ctrls_couplings = []
ctrls_args = []

AGPtype = "chebyshev"
norm_type = "trace"
window_start = 0.25
window_end = 5.0

grid_size = 1000

listof_agp_orders = np.arange(1, 7 + 1)

compare_fid_vs_ord(
    listof_agp_orders,
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    target_symmetries,
    model_kwargs,
    sched,
    ctrls,
    ctrls_couplings,
    ctrls_args,
    AGPtype,
    window_start,
    window_end,
    norm_type,
    grid_size,
)
