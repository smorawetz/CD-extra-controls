import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy
import quspin

sys.path.append(os.environ["CD_CODE_DIR"])

from plots.plot_utils import std_settings

from utils.file_IO import load_data_spec_fn
from utils.file_naming import make_file_name, make_controls_name, make_model_str

from tools.build_ham import build_ham
from tools.symmetries import get_symm_op


CUTOFF = 1e-8
MIN_OMEGA = 0
MAX_OMEGA = 12
NPTS = 1000

SIGMA = 0.1
GS_SIGMA = 0.5


# def Gaussian for broadening
def gaussian(omega, omega0, gamma):  # omega_0 is datapoint
    return (1 / (gamma * np.sqrt(2 * np.pi))) * np.exp(
        -((omega - omega0) ** 2) / (2 * gamma**2)
    )


def interp_spec_fn(
    listof_lamvals,
    Ns,
    model_name,
    H_params,
    symmetries,
    ctrls,
    ctrls_couplings,
    ctrls_args,
    ground_state=False,
):
    file_name = make_file_name(Ns, model_name, H_params, symmetries, ctrls)
    ctrls_name = make_controls_name(ctrls_couplings, ctrls_args)

    # define a grid which which be used for 2d interp, with lamvals on x
    # axis, and different omegas on y axis
    spec_fn_grid = np.zeros((len(listof_lamvals), NPTS))

    # this is the set of points at which omega wiil be plotted
    omega_pts = np.linspace(MIN_OMEGA, MAX_OMEGA, NPTS)

    for j in range(len(lamvals)):
        lamval = lamvals[j]
        omegas, phis = load_data_spec_fn(
            file_name, ctrls_name, lamval, ground_state=ground_state
        )

        sort_omegas = omegas[omegas.argsort()]
        gauss_broad = np.zeros(len(omega_pts))

        omegas = omegas[phis > CUTOFF]
        phis = phis[phis > CUTOFF]

        if ground_state:
            for k in range(len(omegas)):
                omega = omegas[k]
                gauss_broad += phis[k] * gaussian(omega_pts, omega, GS_SIGMA)
        else:
            for k in range(len(omegas)):
                omega = omegas[k]
                gauss_broad += phis[k] * gaussian(omega_pts, omega, SIGMA)

        # need to add normalization
        basis = quspin.basis.spin_basis_general(
            Ns[0],
            **symmetries,
        )
        if not ground_state:  # also divide by Hilbert space dim
            N = basis.Ns
            gauss_broad /= N
        # since partial_\lambda H is extensive, divide by L
        gauss_broad /= Ns[0]

        spec_fn_grid[j, :] = gauss_broad

    # here save in neighbouring data folder to use in interpolation
    np.savetxt(
        "data_dump/spec_fn_data/{0}_lam_data.txt".format(
            make_model_str(Ns, model_name, H_params, ctrls)
        ),
        lamvals,
    )
    np.savetxt(
        "data_dump/spec_fn_data/{0}_omega_data.txt".format(
            make_model_str(Ns, model_name, H_params, ctrls)
        ),
        omega_pts,
    )
    np.savetxt(
        "data_dump/spec_fn_data/{0}_spec_fn_grid.txt".format(
            make_model_str(Ns, model_name, H_params, ctrls)
        ),
        spec_fn_grid,
    )


############# params #############
Ns = [10]
model_name = "NNN_TFIM_1D"
H_params = [1, 0.25, 1]  # seed 0 and disorder strength 0.1
# model_name = "XXZ_1D"
# H_params = [1, 1]  # seed 0 and disorder strength 0.1
boundary_conds = "periodic"

symms = ["translation_1d", "spin_inversion"]
symms_args = [[Ns], [Ns]]
symm_nums = [0, 0]
symmetries = {
    symms[i]: (get_symm_op(symms[i], *symms_args[i]), symm_nums[i])
    for i in range(len(symms))
}
# symmetries["m"] = 0.0
target_symmetries = symmetries

model_kwargs = {}

ctrls = []
ctrls_couplings = []
ctrls_args = []

ground_state = False

lamvals = np.linspace(0, 1, 101)

interp_spec_fn(
    lamvals,
    Ns,
    model_name,
    H_params,
    symmetries,
    ctrls,
    ctrls_couplings,
    ctrls_args,
    ground_state=ground_state,
)
