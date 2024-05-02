import os
import sys

import pickle

sys.path.append(os.environ["CD_CODE_DIR"])

from tools.ham_couplings import sin_coupling, dlam_sin_coupling

couplings_dict = {
    "sin": sin_coupling,
}
dlam_couplings_dict = {
    "sin": dlam_sin_coupling,
}

with open("{0}/dicts/couplings.pkl".format(os.environ["CD_CODE_DIR"]), "wb") as f:
    pickle.dump(couplings_dict, f)


with open("{0}/dicts/dlam_couplings.pkl".format(os.environ["CD_CODE_DIR"]), "wb") as f:
    pickle.dump(dlam_couplings_dict, f)
