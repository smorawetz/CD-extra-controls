import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy

sys.path.append(os.environ["CD_CODE_DIR"])

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
def plot_spec_fn(
    lamval, Ns, model_name, H_params, symmetries, ctrls, ctrls_couplings, ctrls_args
):

    fig, ax = plt.subplots(figsize=(9, 5))

    file_name = make_file_name(Ns, model_name, H_params, symmetries, ctrls)
    ctrls_name = make_controls_name(ctrls_couplings, ctrls_args)

    # omegas = freqs, phis = spec_fn
    omegas, phis = load_data_spec_fn(file_name, ctrls_name, lamval)

    sort_omegas = omegas[omegas.argsort()]
    min_plot_omega = 0
    max_plot_omega = omegas[np.where(omegas > 9)[0][0]]
    omega_pts = np.linspace(min_plot_omega, max_plot_omega, 1000)
    gauss_broad = np.zeros(len(omega_pts))

    for omega in omegas[phis > CUTOFF]:
        gauss_broad += gaussian(omega_pts, omega, 0.01)

    # start_fit = 600
    # end_fit = 990
    # asymp = scipy.stats.linregress(
    #     omega_pts[start_fit:end_fit] * np.log(omega_pts[start_fit:end_fit]),
    #     np.log(gauss_broad[start_fit:end_fit]),
    # )

    ax.plot(omega_pts, np.log(gauss_broad), "r-", linewidth=3)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\log\Phi(\omega)$")
    fig.legend(frameon=False, loc=[0.55, 0.79])
    plt.savefig("plots/images/spectral_function_test.pdf".format())


############# params #############
Ns = [8]
model_name = "TFIM_1D"
H_params = [1, 1]  # seed 0 and disorder strength 0.1
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

tau = 1
sched = SmoothSchedule(tau)  # always use tau = 1 for grid save

lamval = 0.5

plot_spec_fn(
    lamval, Ns, model_name, H_params, symmetries, ctrls, ctrls_couplings, ctrls_args
)
