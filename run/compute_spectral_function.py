import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

from tools.schedules import SmoothSchedule
from tools.symmetries import get_symm_op

from scripts.calc_spectral_function import calc_spectral_function

Ns = [8]
model_name = "TFIM_1D"
H_params = [1, 1]
boundary_conds = "periodic"

symms = ["translation_1d", "spin_inversion"]
symms_args = [[Ns], [Ns]]
symm_nums = [0, 0]
symmetries = {
    symms[i]: (
        get_symm_op(symms[i], *symms_args[i]),
        symm_nums[i],
    )
    for i in range(len(symms))
}

model_kwargs = {}

# schedule will be for coeffs grid, or evolution depending on script
tau = 1.0
sched = SmoothSchedule(tau)

lamval = 0.5

calc_spectral_function(
    lamval,
    model_name,
    Ns,
    H_params,
    boundary_conds,
    symmetries,
    model_kwargs,
    sched,
    ground_state=False,
)
