import os
import sys
import pickle

sys.path.append(os.environ["CD_CODE_DIR"])

import numpy as np
import matplotlib.pyplot as plt
import scipy
import quspin

with open("{0}/dicts/fit_funcs.pkl".format(os.environ["CD_CODE_DIR"]), "rb") as f:
    fit_funcs_dict = pickle.load(f)

from plots.plot_utils import std_settings

from utils.file_IO import load_data_spec_fn
from utils.file_naming import make_file_name, make_controls_name

from tools.build_ham import build_ham
from tools.calc_universal_fit_coeffs import fit_universal_coeffs
from tools.schedules import SmoothSchedule
from tools.symmetries import get_symm_op

std_settings()

CUTOFF = 1e-16


# define gaussian broadening for each point
def gaussian(omega, omega0, gamma):  # omega_0 is datapoint
    return (1 / (gamma * np.sqrt(2 * np.pi))) * np.exp(
        -((omega - omega0) ** 2) / (2 * gamma**2)
    )


def lorentzian(omega, omega0, gamma):  # omega_0 is datapoint
    return 1 / np.pi * (gamma) / ((omega - omega0) ** 2 + (gamma) ** 2)


############# plotting #############
def plot_spec_fn_driven_approx(
    lamval,
    Ns,
    model_name,
    H_params,
    symmetries,
    ctrls,
    ctrls_couplings,
    ctrls_args,
    agp_order,
    AGPtype,
    window_start,
    window_end,
):

    fig, ax = plt.subplots(figsize=(9, 5))

    file_name = make_file_name(Ns, model_name, H_params, symmetries, ctrls)
    ctrls_name = make_controls_name(ctrls_couplings, ctrls_args)

    # omegas = freqs, phis = spec_fn
    omegas, phis = load_data_spec_fn(file_name, ctrls_name, lamval)

    sort_omegas = omegas[omegas.argsort()]
    min_plot_omega = 0
    max_plot_omega = omegas[np.where(omegas > 10)[0][0]]
    omega_pts = np.linspace(min_plot_omega, max_plot_omega, 1000)
    gauss_broad = np.zeros(len(omega_pts))

    omegas = omegas[phis > CUTOFF]
    phis = phis[phis > CUTOFF]

    for j in range(len(omegas)):
        omega = omegas[j]
        gauss_broad += phis[j] * gaussian(omega_pts, omega, 0.2)

    # need to add normalization
    basis = quspin.basis.spin_basis_general(
        Ns[0],
        **symmetries,
    )
    N = basis.Ns
    gauss_broad /= N
    # since partial_\lambda H is extensive, divide by L
    gauss_broad /= Ns[0]

    coeffs = fit_universal_coeffs(agp_order, AGPtype, window_start, window_end)
    fit_func = fit_funcs_dict[AGPtype]
    fit_poly = fit_func(omega_pts / window_end, *coeffs)

    # here make plot without log
    ax.plot(
        omega_pts,
        # gauss_broad * (fit_poly + 1 / omega_pts) ** 2,
        gauss_broad * (fit_poly + 1 / omega_pts) ** 2 / omega_pts**2,
        "r-",
        linewidth=3,
        label="Approx. AGP driving",
    )
    ax.plot(
        omega_pts,
        # gauss_broad * (1 / omega_pts) ** 2,
        gauss_broad * (1 / omega_pts) ** 4,
        "b-",
        linewidth=3,
        label="No driving",
    )

    ax.axhline(1e-1, color="k", linestyle="--", label=r"$\varepsilon = 10^{-1}$")

    ax.set_ylabel(r"$\Phi(\omega)$")
    ax.set_yscale("log")

    ax.set_xlabel(r"$\omega$")
    fig.legend(frameon=False, loc=[0.55, 0.79], fontsize=12)
    plt.savefig("plots/images/driven_spectral_function_N{0}.pdf".format(Ns[0]))


############# params #############
Ns = [12]
model_name = "NNN_TFIM_1D"
H_params = [1, 0.25, 1]  # seed 0 and disorder strength 0.1
boundary_conds = "periodic"

symms = ["translation_1d", "spin_inversion"]
symms_args = [[Ns], [Ns]]
symm_nums = [0, 0]
symmetries = {
    symms[i]: (get_symm_op(symms[i], *symms_args[i]), symm_nums[i])
    for i in range(len(symms))
}
target_symmetries = symmetries

model_kwargs = {}

ctrls = []
ctrls_couplings = []
ctrls_args = []

agp_order = 5
AGPtype = "chebyshev"

opt_ords = np.loadtxt("TFIM_clean_opt_agp_orders.txt")
opt_deltas = np.loadtxt("TFIM_clean_opt_deltas.txt")
ind = np.where(opt_ords == agp_order)[0][0]
window_end = 4.0
# window_end = 6.24
window_start = opt_deltas[ind] * window_end / 4.0

tau = 1
sched = SmoothSchedule(tau)  # always use tau = 1 for grid save

lamval = 0.6

plot_spec_fn_driven_approx(
    lamval,
    Ns,
    model_name,
    H_params,
    symmetries,
    ctrls,
    ctrls_couplings,
    ctrls_args,
    agp_order,
    AGPtype,
    window_start,
    window_end,
)
