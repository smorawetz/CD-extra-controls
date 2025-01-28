import os
import sys

import pickle

sys.path.append(os.environ["CD_CODE_DIR"])

from models.spinhalf_1d.TFIM_annealing_1d import TFIM_Annealing_1D
from models.spinhalf_1d.TFIM_random_annealing_1d import TFIM_Random_Annealing_1D
from models.spinhalf_1d.NNN_TFIM_annealing_1d import NNN_TFIM_Annealing_1D
from models.spinhalf_1d.LTFIM_annealing_1d import LTFIM_Annealing_1D
from models.spinhalf_1d.TFIM_sweep_disorder_1d import TFIM_Sweep_Disorder_1D
from models.spinhalf_1d.LR_Ising_annealing_1d import LR_Ising_Annealing_1D
from models.spinhalf_1d.XXZ_annealing_1d import XXZ_Annealing_1D
from models.spinhalf_1d.XXZ_Heisenberg_annealing_1d import XXZ_Heisenberg_Annealing_1D
from models.spinhalf_1d.XXZ_ToHeisenberg_annealing_1d import (
    XXZ_ToHeisenberg_Annealing_1D,
)
from models.spinhalf_1d.XXZ_NearHeisenberg_annealing_1d import (
    XXZ_NearHeisenberg_Annealing_1D,
)
from models.spinhalf_1d.XXZ_Delta1ToDelta2_annealing_1d import (
    XXZ_Delta1ToDelta2_Annealing_1D,
)
from models.spinhalf_1d.XXZ_Delta2ToDelta1_annealing_1d import (
    XXZ_Delta2ToDelta1_Annealing_1D,
)
from models.spinhalf_1d.XXZ_HalfHeisenberg_annealing_1d import (
    XXZ_HalfHeisenberg_Annealing_1D,
)
from models.spinhalf_1d.local_field_sensing_1d import Local_Field_Sensing_1D
from models.spinhalf_1d.local_field_sensing_1d_sweep import Local_Field_Sensing_1D_Sweep

from models.spinless_fermion_1d.TFIM_k_block_annealing_1d import (
    TFIM_k_Block_Annealing_1D,
)
from models.spinless_fermion_1d.TFIM_cell_random_k_block import TFIM_Cell_Random_k_Block

from models.spinhalf_2d.disorder_Ising_2d import Disorder_Ising_2D

from models.largeS_1d.sensing_squeezing import Sensing_Spin_Squeezing
from models.largeS_1d.sensing_squeezing_endFM import Sensing_Spin_Squeezing_EndFM

models_dict = {
    "TFIM_1D": TFIM_Annealing_1D,
    "TFIM_Random_1D": TFIM_Random_Annealing_1D,
    "NNN_TFIM_1D": NNN_TFIM_Annealing_1D,
    "LTFIM_1D": LTFIM_Annealing_1D,
    "TFIM_Sweep_Disorder_1D": TFIM_Sweep_Disorder_1D,
    "LR_Ising_1D": LR_Ising_Annealing_1D,
    "XXZ_1D": XXZ_Annealing_1D,
    "XXZ_Heisenberg_1D": XXZ_Heisenberg_Annealing_1D,
    "XXZ_ToHeisenberg_1D": XXZ_ToHeisenberg_Annealing_1D,
    "XXZ_HalfHeisenberg_1D": XXZ_HalfHeisenberg_Annealing_1D,
    "XXZ_NearHeisenberg_1D": XXZ_NearHeisenberg_Annealing_1D,
    "XXZ_Delta1ToDelta2_1D": XXZ_Delta1ToDelta2_Annealing_1D,
    "XXZ_Delta2ToDelta1_1D": XXZ_Delta2ToDelta1_Annealing_1D,
    "Field_Sensing_1D": Local_Field_Sensing_1D,
    "Field_Sensing_1D_Sweep": Local_Field_Sensing_1D_Sweep,
    "TFIM_k_Block_Annealing_1D": TFIM_k_Block_Annealing_1D,
    "TFIM_k_Block_Cell_Random": TFIM_Cell_Random_k_Block,
    "Disorder_Ising_2D": Disorder_Ising_2D,
    "Large_S_Sensing": Sensing_Spin_Squeezing,
    "Large_S_Sensing_EndFM": Sensing_Spin_Squeezing_EndFM,
}
param_names_dict = {
    "TFIM_1D": "J{0}_hx{1}",
    "TFIM_Random_1D": "J{0}_hx{1}_Nd{2}",  # J and h influence draw of params
    "NNN_TFIM_1D": "J{0}_J2{1}_hx{2}",
    "LTFIM_1D": "J{0}_hx{1}_hz{2}",
    "TFIM_Sweep_Disorder_1D": "J{0}_hi{1}_hf{2}_disorder{3:.6f}_seed{4}",
    "LR_Ising_1D": "J{0}_hx{1}_alpha{2}",
    "XXZ_1D": "J{0}_Delta{1}",
    "XXZ_Heisenberg_1D": "J{0}_Delta{1}",
    "XXZ_ToHeisenberg_1D": "J{0}_Delta{1}",
    "XXZ_HalfHeisenberg_1D": "J{0}_Delta{1}",
    "XXZ_NearHeisenberg_1D": "J{0}_Delta{1}",
    "XXZ_Delta1ToDelta2_1D": "J{0}_Delta{1}",
    "XXZ_Delta2ToDelta1_1D": "J{0}_Delta{1}",
    "Field_Sensing_1D": "J{0}_hx{1}_hz{2:.6f}",
    "Field_Sensing_1D_Sweep": "J{0}_hx{1}_hz{2:.6f}",
    "TFIM_k_Block_Annealing_1D": "J{0}_hx{1}",  # doesn't include k in file naming
    "TFIM_k_Block_Cell_Random": "J{0}_hx{1}_seed{2}",  # h is to draw from uniform [0, 2 * hx]
    "Disorder_Ising_2D": "J{0}_hx{1}_disorder{2:.6f}_seed{3}",
    "Large_S_Sensing": "chi{0}_h{1}_g{2:.6f}",
    "Large_S_Sensing_EndFM": "chi{0}_h{1}_g{2:.6f}",
}

with open("{0}/dicts/models.pkl".format(os.environ["CD_CODE_DIR"]), "wb") as f:
    pickle.dump(models_dict, f)


with open(
    "{0}/dicts/model_param_names.pkl".format(os.environ["CD_CODE_DIR"]), "wb"
) as f:
    pickle.dump(param_names_dict, f)
