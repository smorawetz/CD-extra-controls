import quspin

# NOTE: Currently on works for global controls. But this might
# not be helpful for e.g. disordered systems. Will reevaluate later


def spin_half_controls(op_size, dim):
    """Gets functions to build extra controls for a spin-half system
    depending on the support of the control operator
    """
    if op_size == 1:
        return build_1site_controls
    elif op_size == 2:
        if dim == 1:
            return build_2site_controls_1d
    elif op_size == 3:
        if dim == 1:
            return build_3site_controls_1d
    else:
        raise ValueError("Do not have code to build controls with support ", op_size)


def build_1site_controls(ham, ctrl, coupling, coupling_args):
    """Build 1-site extra controls term in a given direction with
    some coupling specified by user
    Parameters:
        ham (Hamiltonian_CD):   CD Hamiltonian for system of interest
        ctrl (str):             Direction of the control term
        coupling (function):    Coupling function for the control term
        coupling_args (list):   Arguments for the coupling function
    """
    static = []
    dynamic = [ctrl, [[1, i] for i in range(ham.Ns)], coupling, [*coupling_args]]
    return quspin.operator.hamiltonian(static, dynamic, basis=ham.basis)


def build_2site_controls_1d(ham, ctrl, coupling, *coupling_args):
    """Build 2-site (in 1 dimension) extra controls term in a given
    direction with some coupling specified by user
    Parameters:
        ham (Hamiltonian_CD):   CD Hamiltonian for system of interest
        ctrl (str):             Direction of the control term
        coupling (function):    Coupling function for the control term
        coupling_args (list):   Arguments for the coupling function
    """
    static = []
    dynamic = [
        ctrl,
        [[1, *self.pairs[i]] for i in range(len(self.pairs))],
        coupling,
        [*coupling_args],
    ]
    return quspin.operator.hamiltonian(static, dynamic, basis=ham.basis)


def build_3site_controls_1d(ham, ctrl, coupling, *coupling_args):
    """Build 3-site (in 1 dimension) extra controls term in a given
    direction with some coupling specified by user
    Parameters:
        ham (Hamiltonian_CD):   CD Hamiltonian for system of interest
        ctrl (str):             Direction of the control term
        coupling (function):    Coupling function for the control term
        coupling_args (list):   Arguments for the coupling function
    """
    static = []
    dynamic = [
        ctrl,
        [[1, *self.triplets[i]] for i in range(len(self.triplets))],
        coupling,
        [*coupling_args],
    ]
    return quspin.operator.hamiltonian(static, dynamic, basis=ham.basis)
