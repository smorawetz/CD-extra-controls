import os
import sys

import pickle

sys.path.append(os.environ["CD_CODE_DIR"])

from models.spinhalf_1d.TFIM_annealing_1d import TFIM_Annealing_1D
from models.spinhalf_1d.TFIM_sweep_disorder_1d import TFIM_Sweep_Disorder_1D
from models.spinhalf_1d.LR_Ising_annealing_1d import LR_Ising_Annealing_1D
from models.spinhalf_1d.local_field_sensing_1d import Local_Field_Sensing_1D
from models.spinhalf_1d.local_field_sensing_1d_sweep import Local_Field_Sensing_1D_Sweep

models_dict = {
    "TFIM_1D": TFIM_Annealing_1D,
    "TFIM_Sweep_Disorder_1D": TFIM_Sweep_Disorder_1D,
    "LR_Ising_1D": LR_Ising_Annealing_1D,
    "Field_Sensing_1D": Local_Field_Sensing_1D,
    "Field_Sensing_1D_Sweep": Local_Field_Sensing_1D_Sweep,
}
param_names_dict = {
    "TFIM_1D": "J{0}_hx{1}",
    "TFIM_Sweep_Disorder_1D": "J{0}_hi{1}_hf{2}_disorder{3:.6f}_seed{4}",
    "LR_Ising_1D": "J{0}_hx{1}_alpha{2}",
    "Field_Sensing_1D": "J{0}_hx{1}_hz{2:.6f}",
    "Field_Sensing_1D_Sweep": "J{0}_hx{1}_hz{2:.6f}",
}

with open("{0}/dicts/models.pkl".format(os.environ["CD_CODE_DIR"]), "wb") as f:
    pickle.dump(models_dict, f)


with open(
    "{0}/dicts/model_param_names.pkl".format(os.environ["CD_CODE_DIR"]), "wb"
) as f:
    pickle.dump(param_names_dict, f)
