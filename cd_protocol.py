import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

from tools.matrix_evolution import do_evolution


class CD_Protocol:
    """This is a "CD" protocol, which includes all the information about
    the Hamiltonian, any extra controls, and the necessary information to
    construct the AGP (coeffs, AGPtype, etc.)"""

    def __init__(
        self,
        ham,
        AGPtype,
        ctrls,
        couplings,
        couplings_args,
        schedule,
        grid_size,
    ):
        """
        Parameters:
            ham (Hamiltonian_CD):           Counterdiabatic Hamiltonian of interest
            AGPtype (str):                  Type of approximate AGP to construct depending
                                            on the type of AGP desired
            ctrls (list):                   List of control Hamiltonians
            couplings (list):               List of coupling functions for control terms
            couplings_args (list):          List of list of arguments for the
                                            coupling functions
            schedule (Schedule):            Schedule object that encodes $\lambda(t)$
            grid_size (int):                Number of time steps to take
        """
        self.ham = ham
        self.AGPtype = AGPtype
        self.ctrls = ctrls
        self.couplings = couplings
        self.couplings_args = couplings_args
        self.schedule = schedule
        self.grid_size = grid_size

    def matrix_evolve(self, init_state, wfs_fname, save_states=False):
        """Evolve `init_state` in accordance with the CD protocol
        Parameters:
            init_state (np.array):      Initial state to evolve
            wfs_fname (str):            String to save the wavefunctions
            save_states (bool):         Whether to save the wavefunctions
        Returns:
            final_state (np.array):     Final state after evolution
        """
        return do_evolution(
            self.ham,
            wfs_fname,
            self.AGPtype,
            self.ctrls,
            self.couplings,
            self.couplings_args,
            self.grid_size,
            init_state,
            save_states=save_states,
        )
