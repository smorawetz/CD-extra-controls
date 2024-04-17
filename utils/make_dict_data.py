import os
import sys

import pickle

sys.path.append(os.environ["CD_CODE_DIR"])

from models.TFIM_annealing_1d import TFIM_Annealing_1D

models_dict = {"TFIM_1D": TFIM_Annealing_1D}
param_names_dict = {
    "TFIM_1D": "J{0}_hx{1}",
}

with open("{0}/dicts/models.pkl".format(os.environ["CD_CODE_DIR"]), "wb") as f:
    pickle.dump(models_dict, f)


with open(
    "{0}/dicts/model_param_names.pkl".format(os.environ["CD_CODE_DIR"]), "wb"
) as f:
    pickle.dump(param_names_dict, f)
