import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

from tools.schedules import LinearSchedule, SmoothSchedule
from tools.symmetries import get_symm_op
from utils.file_naming import make_base_fname

from scripts.calc_commutator_coeffs import calc_comm_coeffs
from scripts.calc_krylov_coeffs import calc_kry_coeffs
from scripts.iterative_GS_optimization import do_iterative_evolution
from scripts.optimize_harmonics_coeffs import optim_harmonic_coeffs
from scripts.time_evolution import run_time_evolution

# define the various parameters of the model/task
Ns = 4
coeffs_model_name = "TFIM_Disorder_1D"
coeffs_H_params = [1, 1, 0.1, 1]  # seed 1 and disorder strength 0.1
coeffs_boundary_conds = "periodic"

symms = []  # break Z2 and translational symmetry
symms_args = []
symm_nums = []
coeffs_symmetries = {
    symms[i]: (get_symm_op(symms[i], *symms_args[i]), symm_nums[i])
    for i in range(len(symms))
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

agp_order = 1
AGPtype = "commutator"
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

# now define the kwargs which are specific to each script
append_str = "std"

kwargs = {"append_str": append_str}

calc_kry_coeffs(*args, **kwargs)

Ns = 4
evolve_model_name = "TFIM_1D"
evolve_H_params = [1, 1]
evolve_boundary_conds = "periodic"

symms = ["translation_1d", "spin_inversion"]
symms_args = [[Ns], [Ns]]
symm_nums = [0, 0]
evolve_symmetries = {
    symms[i]: (get_symm_op(symms[i], *symms_args[i]), symm_nums[i])
    for i in range(len(symms))
}
evolve_target_symmetries = evolve_symmetries

args = (
    ## H params
    Ns,
    evolve_model_name,
    evolve_H_params,
    evolve_boundary_conds,
    evolve_symmetries,
    evolve_target_symmetries,
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
)

coeffs_fname = (
    make_base_fname(
        Ns,
        coeffs_model_name,
        coeffs_H_params,
        coeffs_symmetries,
        ctrls,
        agp_order,
        AGPtype,
        norm_type,
        grid_size,
        coeffs_sched,
        append_str,
    )
    + "_coeffs"
)

kwargs = {
    "save_wf": True,
    "coeffs_fname": coeffs_fname,
    "coeffs_sched": coeffs_sched,
    "wfs_save_append_str": append_str,
    "print_fid": True,
}

run_time_evolution(*args, **kwargs)
