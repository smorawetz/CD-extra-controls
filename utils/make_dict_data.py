param_names_dict = {
    "TFIM_1D": "J{0}_hx{1}",
}

with open("../dicts/model_param_names.pkl", "wb") as f:
    pickle.dump(param_names_dict, f)
