import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

from tools.schedules import LinearSchedule, SmoothSchedule
from tools.symmetries import get_symm_op

from scripts.calc_commutator_coeffs import calc_comm_coeffs
from scripts.calc_krylov_coeffs import calc_kry_coeffs

# define the various parameters of the model/task
Ns = [4]
coeffs_model_name = "TFIM_1D"
coeffs_H_params = [1, 1]  # seed 1 and disorder strength 0.1
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

agp_order = 1
# AGPtype = "commutator"
AGPtype = "krylov"
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

kwargs = {}

# calc_comm_coeffs(*args, **kwargs)
calc_kry_coeffs(*args, **kwargs)
