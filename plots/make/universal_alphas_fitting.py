import os
import sys

import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy

sys.path.append(os.environ["CD_CODE_DIR"])

from plots.plot_utils import std_settings

from tools.build_ham import build_ham
from tools.calc_universal_fit_coeffs import fit_universal_coeffs
from tools.schedules import SmoothSchedule
from tools.symmetries import get_symm_op
from utils.file_naming import make_base_fname

std_settings()


############# plotting #############
def plot_alphas_fitting(
    tau_frac,  # what fraction of way through protocol to plot
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    target_symmetries,
    ctrls,
    agp_order,
    window_start,
    window_end,
    norm_type,
    grid_size,
    sched,
    append_str,
    load_alphas=False,
):
    tval = tau * tau_frac  # when to plot
    fname = make_base_fname(
        Ns,
        model_name,
        H_params,
        symmetries,
        ctrls,
        agp_order,
        "commutator",  # fitting only sensible for alphas
        norm_type,
        grid_size,
        sched,
        append_str,
    )
    # employ CD Hamiltonian to for excitation frequencies, possibly alphas
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

    alphas = fit_universal_coeffs(agp_order, window_start, window_end)

    # now get the excitation frequencies directly from the model at a given t
    eigvals, eigvecs = ham.build_bare_H().eigh(time=tval)
    dlamH = ham.build_dlam_H()
    freqs = []
    for m in range(len(eigvals)):
        for n in range(m + 1, len(eigvals)):
            mtx_elt = dlamH.matrix_ele(eigvecs[:, m], eigvecs[:, n])
            freq = eigvals[n] - eigvals[m]
            if np.abs(mtx_elt) > 1e-12:
                freqs.append(freq)
    freqs = np.array(freqs)

    x = np.linspace(1e-3, 1.2 * max(freqs), 1000)
    y = np.zeros_like(x)
    for n in range(len(alphas)):
        y += -alphas[n] * x ** (2 * n + 1)  # - since flipped

    fig, ax = plt.subplots(figsize=(9, 5))
    # ax.set_yscale("log")
    ax.set_ylim(0, 10)
    ax.plot(x, 1 / x, "k--", linewidth=2, label=r"$1/\omega$")
    ax.plot(freqs, 1 / freqs, "ro", markersize=7, label=r"Excitations")
    ax.plot(x, y, "b-", linewidth=2, label=r"Fit at order {0}".format(agp_order))
    ax.set_xlabel(r"$\omega$")
    fig.legend(frameon=False, loc=[0.61, 0.65])
    plt.savefig(
        "plots/images/universal_alphas_fitting_frac{0}_{1}.pdf".format(tau_frac, fname)
    )


############# params #############
g = 0.00
Ns = 10
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

# model_kwargs = {"disorder_strength": 1, "disorder_seed": 0}
model_kwargs = {"disorder_strength": 0, "disorder_seed": 0}

ctrls = []

agp_order = 10
window_start = 0.5
window_end = 4.0
norm_type = "trace"

grid_size = 1000

tau = 1
sched = SmoothSchedule(tau)  # always use tau = 1 for grid save

append_str = "std"

tau_frac = 0.5
plot_alphas_fitting(
    tau_frac,
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    target_symmetries,
    ctrls,
    agp_order,
    window_start,
    window_end,
    norm_type,
    grid_size,
    sched,
    append_str,
    load_alphas=False,
)
