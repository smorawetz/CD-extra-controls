import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy

from matplotlib.ticker import MaxNLocator

sys.path.append(os.environ["CD_CODE_DIR"])

from plots.plot_utils import std_settings
from tools.calc_universal_fit_coeffs import fit_universal_coeffs
from tools.build_ham import build_ham
from tools.lin_alg_calls import calc_fid
from tools.schedules import SmoothSchedule
from tools.symmetries import get_symm_op
from utils.file_naming import make_fit_coeffs_fname
from utils.grid_utils import get_universal_coeffs_func
from scripts.time_evolution_universal_coeffs import run_time_evolution_universal

std_settings()


############# plotting #############
def plot_fid_vs_universal_fit_order(
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
    window_start,
    window_end,
    AGPtype,
    norm_type,
    grid_size,
    sched,
    append_str,
):

    rescale = 1 / window_end
    # rescale = 1

    fids = []
    for agp_order in listof_agp_orders:
        coeffs = fit_universal_coeffs(agp_order, AGPtype, window_start, window_end)
        coeffs_fname = make_fit_coeffs_fname(
            AGPtype, agp_order, window_start, window_end
        )

        np.savetxt(
            "{0}/coeffs_data/{1}.txt".format(os.environ["CD_CODE_DIR"], coeffs_fname),
            coeffs,
        )

        args = (
            ## H params
            Ns,
            model_name,
            H_params,
            boundary_conds,
            symmetries,
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
            rescale,
        )

        # now load coefficients for the universal fitting

        kwargs = {
            "save_wf": True,
            "coeffs_fname": coeffs_fname,
            "wfs_save_append_str": append_str,
            "print_fid": True,
        }

        # print("H build ", agp_order)

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
            rescale=rescale,
        )

        # this is particularized to the alphas fit
        ham.alphas_interp = get_universal_coeffs_func(coeffs)

        # this is particularized to cheby
        ham.polycoeffs_interp = get_universal_coeffs_func(coeffs)

        targ_state = ham.get_targ_gstate()

        final_wf = run_time_evolution_universal(*args, **kwargs)

        fids.append(calc_fid(final_wf, targ_state))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(listof_agp_orders, fids, "ko")
    ax.set_xlabel("AGP order")
    ax.set_ylabel(r"$\mathcal{F}$")
    plt.savefig("plots/images/fid_vs_universal_fit_order.pdf")


############# params #############
g = 0.1

# define the various parameters of the model/task
Ns = 4
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
target_symmetries = symmetries

model_kwargs = {}

# schedule will be for coeffs grid, or evolution depending on script
evolve_tau = 0.001
coeffs_tau = 1
evolve_sched = SmoothSchedule(evolve_tau)
coeffs_sched = SmoothSchedule(coeffs_tau)

ctrls = []
ctrls_couplings = []
ctrls_args = []

agp_order = 9
# AGPtype = "commutator"
AGPtype = "chebyshev"
norm_type = "trace"
window_start = 0.1
window_end = 8.0

grid_size = 1000
append_str = "universal_g{0:.6f}".format(g)

listof_agp_orders = range(1, 31)

plot_fid_vs_universal_fit_order(
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
    window_start,
    window_end,
    AGPtype,
    norm_type,
    grid_size,
    evolve_sched,
    append_str,
)
