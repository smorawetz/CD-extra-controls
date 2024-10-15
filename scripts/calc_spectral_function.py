import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

from tools.build_ham import build_ham
from utils.file_IO import (
    save_data_spec_fn,
    load_raw_data_spec_fn,
    merge_data_spec_fn,
    load_data_spec_fn,
)
from utils.file_naming import make_file_name, make_controls_name


def calc_spectral_function(
    lamval,
    model_name,
    Ns,
    H_params,
    boundary_conds,
    symmetries,
    model_kwargs,
    sched,
    ground_state=False,
):
    ham = build_ham(
        model_name,
        Ns,
        H_params,
        boundary_conds,
        model_kwargs,
        0,  # no agp
        "trace",  # doesn't matter if no agp
        sched,
        symmetries={},
        target_symmetries={},
    )
    tval = sched.get_t(lamval)
    freqs, spec_fn = ham.get_spectral_function(tval, ground_state=ground_state)
    file_name = make_file_name(Ns, model_name, H_params, symmetries, [])  # [] = no ctrl
    controls_name = make_controls_name([], [])  # assuming no extra controls
    save_data_spec_fn(file_name, controls_name, freqs, spec_fn, lamval)
    return freqs, spec_fn
