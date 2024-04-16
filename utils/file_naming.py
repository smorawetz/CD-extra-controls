with open("../dicts/model_param_names.pkl", "rb") as f:
    param_names_dict = pickle.load(f)


def make_H_params_str(model_name, H_params):
    return param_names_dict[model_name].format(*H_params)


def make_controls_str(ctrls):
    if ctrls is not None:
        return "_".join(ctrls)
    else:
        return "no_controls"


def make_agp_str(AGPtype, agp_order):
    if agp_order == 0:
        return "no_agp"
    if AGPtype == "commutator":
        return f"commutator_{agp_order}agp"
    elif AGPtype == "krylov":
        return f"krylov_{agp_order}agp"
    else:
        raise ValueError(f"AGPtype {agp_order} not recognized")


def make_model_name_str(ham, model_name, ctrls, AGPtype):
    H_params_str = make_H_params_str(model_name, ham.H_params)
    ctrls_str = make_controls_str(ctrls)
    agp_str = make_agp_str(AGPtype, ham.agp_order)
    return f"{model_name}_{H_params_str}_{ctrls_str}_{agp_str}"
