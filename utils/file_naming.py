import os

import pickle

env_dir = os.environ["CD_CODE_DIR"]
with open(f"{env_dir}/dicts/model_param_names.pkl", "rb") as f:
    param_names_dict = pickle.load(f)

norm_type_dict = {"trace": "infT", "ground_state": "zeroT"}
AGPtype_dict = {"commutator": "comm", "krylov": "kry"}
symmetries_names_dict = {"translation_1d": "K{0}", "spin_inversion": "Z{0}"}


def make_H_params_str(model_name, H_params):
    return param_names_dict[model_name].format(*H_params)


def make_controls_str(ctrls):
    if ctrls != []:
        return "_".join(ctrls)
    else:
        return "no_controls"


def make_agp_str(AGPtype, norm_type, agp_order):
    if agp_order == 0:
        return "no_agp"
    if AGPtype in AGPtype_dict.keys() and norm_type in norm_type_dict.keys():
        return f"{AGPtype_dict[AGPtype]}_{norm_type_dict[norm_type]}_ord{agp_order}agp"
    elif AGPtype not in AGPtype_dict.keys():
        raise ValueError(f"AGPtype {agp_order} not recognized")
    elif norm_type not in norm_type_dict.keys():
        raise ValueError(f"norm_type {norm_type} not recognized")


def make_model_name_str(ham, model_name, ctrls, AGPtype, norm_type):
    H_params_str = make_H_params_str(model_name, ham.H_params)
    ctrls_str = make_controls_str(ctrls)
    agp_str = make_agp_str(AGPtype, norm_type, ham.agp_order)
    return f"{model_name}_N{ham.Ns}_{H_params_str}_{ctrls_str}_{agp_str}"


def make_symmetries_str(symmetries):
    if symmetries == []:
        return "no_symm"
    symm_strs = []
    for symmetry in symmetries:  # is a dict of (key: (op, num))
        symm_strs.append(
            symmetries_names_dict[symmetry].format(symmetries[symmetry][1])
        )
    return "_".join(symm_strs)


def make_coeffs_fname(
    ham,
    model_name,
    ctrls,
    AGPtype,
    norm_type,
    grid_size,
    tau,
    append_str,
):
    model_name_str = make_model_name_str(ham, model_name, ctrls, AGPtype, norm_type)
    agp_str = make_agp_str(AGPtype, norm_type, ham.agp_order)
    return f"{model_name_str}_{grid_size}steps_tau{tau}_{append_str}_coeffs"
