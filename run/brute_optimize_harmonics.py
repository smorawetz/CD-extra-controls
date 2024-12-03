import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import numpy as np

from cd_protocol import CD_Protocol
from tools.build_ham import build_ham
from tools.lin_alg_calls import calc_fid
from tools.schedules import LinearSchedule, SmoothSchedule
from tools.symmetries import get_symm_op

from utils.file_IO import (
    save_data_agp_coeffs,
    save_data_optimization_fids,
    save_data_evolved_wfs,
)
from utils.file_naming import make_data_dump_name, make_controls_name_no_coeffs

from scripts.optimize_harmonics_coeffs import (
    optim_harmonic_coeffs,
    optim_harmonic_coeffs_line,
)

# define the various parameters of the model/task
Ns = [8]
model_name = "TFIM_1D"
H_params = [1, 1]  # seed 1 and disorder strength 0.1
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

# schedule will be for coeffs grid, or evolution depending on script
tau = 0.01
sched = SmoothSchedule(tau)

ctrls = ["Hc1", "Hc2"]
ctrls_couplings = ["sin", "sin"]
ctrls_harmonics = [1]

# TODO: input scalar here

nonzero_scalars = np.logspace(-4, 0, 41)
scalars = np.append(np.array([0]), nonzero_scalars)

scalar = scalars[1]

coeffs = scalar * np.array([1, -1])
# set coefficients in given instance
ctrls_args = []
for i in range(len(ctrls)):
    ctrls_args.append([sched, *np.append(ctrls_harmonics, coeffs[i :: len(ctrls)])])

agp_order = 0
AGPtype = "krylov"
norm_type = "trace"

grid_size = 1000

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
    tau,
    sched,
    ## controls params
    ctrls,
    ctrls_couplings,
    ctrls_harmonics,
    ## agp params
    agp_order,
    AGPtype,
    norm_type,
    ## simulation params
    grid_size,
)

append_str = "std"
maxfields = 3

kwargs = {"append_str": append_str, "maxfields": maxfields}

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
)

ham.init_controls(ctrls, ctrls_couplings, ctrls_args)

fname_args_dict = {
    "Ns": Ns,
    "model_name": model_name,
    "H_params": H_params,
    "symmetries": symmetries,
    "sched": sched,
    "ctrls": ctrls,
    "ctrls_couplings": ctrls_couplings,
    "ctrls_args": None,  # will be replaced when coeffs are filled in
    "agp_order": agp_order,
    "AGPtype": AGPtype,
    "norm_type": norm_type,
    "grid_size": grid_size,
}

# get coefficients
if agp_order == 0:
    pass
elif AGPtype == "krylov":
    tgrid, lanc_grid = calc_lanc_coeffs_grid(
        ham,
        grid_size,
        sched,
        agp_order,
        norm_type,
        gs_func=None,
    )
    ham.lanc_interp = get_coeffs_interp(sched, sched, tgrid, lanc_grid)  # same sched
    tgrid, gammas_grid = calc_gammas_grid(
        ham,
        grid_size,
        sched,
        agp_order,
        norm_type,
        gs_func=None,
    )
    ham.gammas_interp = get_coeffs_interp(sched, sched, tgrid, gammas_grid)
elif AGPtype == "commutator":
    tgrid, alphas_grid = calc_alphas_grid(
        ham,
        grid_size,
        sched,
        agp_order,
        norm_type,
        gs_func=None,
    )
    ham.alphas_interp = get_coeffs_interp(sched, sched, tgrid, alphas_grid)

cd_protocol = CD_Protocol(
    ham, AGPtype, ctrls, ctrls_couplings, ctrls_args, sched, grid_size
)

init_state = ham.get_init_gstate()
targ_state = ham.get_targ_gstate()

t_data, wf_data = cd_protocol.matrix_evolve(init_state)
final_wf = wf_data[-1, :]
fid = calc_fid(targ_state, final_wf)

# now save relevant data
fname_args_dict["ctrls_args"] = ctrls_args
save_dirname = "{0}/data_dump".format(os.environ["CD_CODE_DIR"])
file_name, protocol_name, _ = make_data_dump_name(*fname_args_dict.values())
# overwrite last part of names list since don't include coeffs magnitude in fname
ctrls_name = make_controls_name_no_coeffs(ctrls_couplings, ctrls_args)
names_list = (file_name, protocol_name, ctrls_name)

if agp_order > 0:
    save_data_agp_coeffs(*names_list, tgrid, gammas_grid, lanc_grid=lanc_grid)
save_data_optimization_fids(*names_list, coeffs, fid)
save_data_evolved_wfs(*names_list, final_wf, tgrid=None, full_wf=None)
