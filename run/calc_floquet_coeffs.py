import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.environ["CD_CODE_DIR"])

from plots.plot_utils import std_settings

from tools.schedules import SmoothSchedule
from tools.symmetries import get_symm_op

from scripts.calc_floquet_coeffs import get_floquet_coeffs

mpl_colors = std_settings()

with open("{0}/dicts/fit_funcs.pkl".format(os.environ["CD_CODE_DIR"]), "rb") as f:
    fit_funcs_dict = pickle.load(f)

# define the various parameters of the model/task
Ns = [10]
coeffs_model_name = "NNN_TFIM_1D"
coeffs_H_params = [1, 0.25, 1]  # seed 1 and disorder strength 0.1
coeffs_boundary_conds = "periodic"

coeffs_symms = ["translation_1d", "spin_inversion"]
coeffs_symms_args = [[Ns], [Ns]]
coeffs_symm_nums = [0, 0]
coeffs_symmetries = {
    coeffs_symms[i]: (
        get_symm_op(coeffs_symms[i], *coeffs_symms_args[i]),
        coeffs_symm_nums[i],
    )
    for i in range(len(coeffs_symms))
}
coeffs_target_symmetries = coeffs_symmetries

model_kwargs = {}

# schedule will be for coeffs grid, or evolution depending on script
evolve_tau = 0.01
coeffs_tau = 1
evolve_sched = SmoothSchedule(evolve_tau)
coeffs_sched = SmoothSchedule(coeffs_tau)

ctrls = []
ctrls_couplings = []
ctrls_args = []

agp_order = 7
AGPtype = "floquet"
norm_type = "trace"

grid_size = 1000

# have generic list of args that get used for every function
args = (
    ## H params
    Ns,
    coeffs_model_name,
    coeffs_H_params,
    coeffs_boundary_conds,
    coeffs_symmetries,
    coeffs_target_symmetries,
    model_kwargs,
    ## schedule params
    coeffs_tau,
    coeffs_sched,
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
)

mu = 1.0
omega0 = 1.0
spec_fn_Ns = [10]

kwargs = {
    "mu": mu,
    "omega0": omega0,
    "spec_fn_Ns": spec_fn_Ns,
}

# now experiment with different values of mu and omega0, and see what happens

tgrid, betas_grid = get_floquet_coeffs(*args, **kwargs)

x = np.linspace(mu, 10, 1000)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(
    x, fit_funcs_dict["bessel"](x / omega0, *betas_grid[grid_size // 2, :]), linewidth=3
)
ax.plot(x, -1 / x, "k--", linewidth=3)
ax.set_ylim(-25, 25)
plt.savefig("test_fit.png")
