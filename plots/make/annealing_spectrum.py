import os
import sys

import csv
import pickle

import numpy as np
import matplotlib.pyplot as plt
import scipy

sys.path.append(os.environ["CD_CODE_DIR"])

from plots.plot_utils import std_settings

from tools.build_ham import build_ham
from tools.calc_universal_fit_coeffs import fit_universal_coeffs
from tools.schedules import SmoothSchedule
from tools.symmetries import get_symm_op

with open("{0}/dicts/fit_funcs.pkl".format(os.environ["CD_CODE_DIR"]), "rb") as f:
    fit_funcs_dict = pickle.load(f)

std_settings()


############# plotting #############
def plot_anneal_spectrum(
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    target_symmetries,
    ctrls,
    grid_size,
    sched,
):
    ham = build_ham(
        model_name,
        Ns,
        H_params,
        boundary_conds,
        model_kwargs,
        0,  # AGP irrelevant for spectrum
        None,  # norm type irrelevant
        sched,
        symmetries=symmetries,
        target_symmetries=symmetries,
        rescale=rescale,
    )
    lamvals = np.linspace(0, 1, grid_size)
    eigvals = np.zeros((len(lamvals), ham.bareH.basis.Ns))
    for i in range(len(lamvals)):
        lam = lamvals[i]
        tval = sched.get_t(lam)
        eigvals[i, :] = ham.build_bare_H().eigvalsh(time=tval)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(ham.bareH.basis.Ns):
        ax.plot(lamvals, eigvals[:, i], linewidth=2, label=r"$n = {0}$".format(i))
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$E_n$")
    fig.legend(frameon=False, loc=[0.69, 0.55])
    plt.savefig("plots/images/spectrum_during_annealing.pdf".format())


############# params #############
full_Ns = 10
Ns = [2]
model_name = "TFIM_k_Block_Annealing_1D"
H_params = [1, 1, np.pi / full_Ns]
boundary_conds = "periodic"

# symms = ["translation_1d", "spin_inversion"]
symms = []
symms_args = [[Ns], [Ns]]
symm_nums = [0, 0]
symmetries = {
    symms[i]: (get_symm_op(symms[i], *symms_args[i]), symm_nums[i])
    for i in range(len(symms))
}
target_symmetries = symmetries

model_kwargs = {}

# schedule will be for coeffs grid, or evolution depending on script
tau = 0.01
sched = SmoothSchedule(tau)

ctrls = []
ctrls_couplings = []
ctrls_args = []

agp_order = 5
AGPtype = "chebyshev"
norm_type = "trace"
window_start = 0.5
window_end = 4.0
rescale = 1 / window_end

grid_size = 1000

plot_anneal_spectrum(
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    target_symmetries,
    ctrls,
    grid_size,
    sched,
)
