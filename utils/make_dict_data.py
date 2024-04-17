from ..models.spinhalf_1d.TFIM_annealing_1d import TFIM_1D

models_dict = {"TFIM_1D": TFIM_1D}
param_names_dict = {
    "TFIM_1D": "J{0}_hx{1}",
}

with open("../dicts/models.pkl", "wb") as f:
    pickle.dump(models_dict, f)


with open("../dicts/model_param_names.pkl", "wb") as f:
    pickle.dump(param_names_dict, f)
