import os
import sys
import pickle

sys.path.append(os.environ["CD_CODE_DIR"])

import numpy as np
import quspin

with open("{0}/dicts/fit_funcs.pkl".format(os.environ["CD_CODE_DIR"]), "rb") as f:
    fit_funcs_dict = pickle.load(f)

from tools.calc_universal_fit_coeffs import fit_universal_coeffs
from tools.schedules import SmoothSchedule
from tools.symmetries import get_symm_op
from utils.file_naming import make_file_name, make_controls_name
from utils.file_IO import load_data_spec_fn, save_data_opt_windows

CUTOFF = 1e-16


# define gaussian broadening for each point
def gaussian(omega, omega0, gamma):  # omega_0 is datapoint
    return (1 / (gamma * np.sqrt(2 * np.pi))) * np.exp(
        -((omega - omega0) ** 2) / (2 * gamma**2)
    )


def calc_max_cost(
    omegas,
    phis,
    agp_order,
    AGPtype,
    window_start,
    window_end,
    lamdot,
):
    coeffs = fit_universal_coeffs(agp_order, AGPtype, window_start, window_end)
    fit_func = fit_funcs_dict[AGPtype]
    fit_poly = fit_func(omegas / window_end, *coeffs)

    cost = lamdot**2 * phis * (fit_poly + 1 / omegas) ** 2 / omegas**2
    # cost = phis * (fit_poly + 1 / omegas) ** 2 / omegas**2
    return np.max(cost)


def find_window(
    lamval,
    omega0,  # single particle max excitation
    eps,  # maximum allowed from cost
    Ns,
    model_name,
    H_params,
    symmetries,
    ctrls,
    ctrls_couplings,
    ctrls_args,
    agp_order,
    AGPtype,
    sched,
):
    # load small size spectral function ED data
    file_name = make_file_name(Ns, model_name, H_params, symmetries, ctrls)
    ctrls_name = make_controls_name(ctrls_couplings, ctrls_args)

    tval = sched.get_t(lamval)
    lamdot = sched.get_lamdot(lamval)
    # load data and filter by frequencies outside single-particle window
    omegas, phis = load_data_spec_fn(file_name, ctrls_name, lamval)

    # cutoff things below single particle energy
    phis = phis[omegas > omega0]
    omegas = omegas[omegas > omega0]
    # cutoff things with matrix elements within machine precision of zero
    omegas = omegas[phis > CUTOFF]
    phis = phis[phis > CUTOFF]

    omega_pts = np.linspace(min(omegas), max(omegas), 1000)
    gauss_broad = np.zeros(len(omega_pts))

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

    # find optimal fitting range for clean TFIM before breaking integrability
    opt_ords = np.loadtxt("TFIM_clean_opt_agp_orders.txt")
    opt_deltas = np.loadtxt("TFIM_clean_opt_deltas.txt")
    ind = np.where(opt_ords == agp_order)[0][0]
    clean_window_start = opt_deltas[ind]
    clean_window_end = omega0

    # slowly increment upper window until epsilon condition is reached
    for window_end in np.arange(clean_window_end, 3 * clean_window_end + 0.1, 0.1):
        window_start = clean_window_start / clean_window_end * window_end
        cost = calc_max_cost(
            omega_pts,
            gauss_broad,
            agp_order,
            AGPtype,
            window_start,
            window_end,
            lamdot,
        )
        if cost < eps:
            break

    window_arr = np.array([window_start, window_end])
    save_data_opt_windows(file_name, ctrls_name, window_arr, lamval)
    return window_start, window_end


# now actually run
Ns = [8]
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

ctrls = []
ctrls_couplings = []
ctrls_args = []

agp_order = 5
AGPtype = "chebyshev"

omega0 = 4.0
EPS = 1e-1

Nlamvals = 101
lamvals = np.linspace(0, 1, Nlamvals)

tau = 1
sched = SmoothSchedule(tau)  # always use tau = 1 for grid save

for lamval in lamvals:
    max_window_start, max_window_end = find_window(
        lamval,
        omega0,
        EPS,
        Ns,
        model_name,
        H_params,
        symmetries,
        ctrls,
        ctrls_couplings,
        ctrls_args,
        agp_order,
        AGPtype,
        sched,
    )
