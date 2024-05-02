import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy

from matplotlib.ticker import MaxNLocator

sys.path.append(os.environ["CD_CODE_DIR"])

from plots.plot_utils import std_settings
from tools.schedules import SmoothSchedule
from tools.symmetries import get_symm_op
from utils.file_naming import make_base_fname

mpl_colors = std_settings()  # use common plot settings, including science plots style


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

ctrls = []

AGPtype = "krylov"
norm_type = "ground_state"

grid_size = 1000
tau = 0.01
sched = SmoothSchedule(tau)

append_str = "iterative"

agp_orders = [1, 3, 5, 7]

# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(4):
    agp_order = agp_orders[i]
    fname = make_base_fname(
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
    fids = np.loadtxt("plots/data/{0}_fids.txt".format(fname))

    fit_start, fit_end = 4, 7
    fit_pts = np.arange(fit_start, fit_end)
    asymp = scipy.stats.linregress(fit_pts, (1 - np.array(fids)[fit_start:fit_end]))

    # ax = axs[i // 2, i % 2]
    ax.plot(
        1 - np.array(fids)[: fit_end + 1],
        "o",
        color=mpl_colors[i],
        markersize=7,
        label=r"$\ell = {0}$".format(agp_order),
    )
    ax.plot(
        asymp.slope * np.arange(fit_end + 1) + asymp.intercept,
        "--",
        color=mpl_colors[i],
    )
    ax.set_yscale("log")
    if i % 2 == 0:
        ax.set_ylabel(r"$1-\mathcal{F}$")
    if i // 2 == 1:
        ax.set_xlabel("Iteration number")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

fig.legend(frameon=False, loc=[0.69, 0.21])
plt.savefig(f"fids_vs_iter_{AGPtype}_compare_convergence.pdf")
