import numpy as np


def sin_coupling(t, sched, *args):
    """Given some variational coefficients and the mode number they are
    associated with, return the sinusoidal coupling at a given time
    Parameters:
        t (float):          Time at which to evaluate the coupling
        sched (Schedule):   Schedule object for the lambda schedule
        args (list):        First half of args are integers, respresenting
                            the harmonic number. Second half are floats,
                            represent the coefficients. Need this hacky
                            workaround because QuSpin hashes dynamic
                            parts with arguments of coupling function
    """
    args = np.array(args)
    ns = args[: len(args) // 2]
    coeffs = args[len(args) // 2 :]
    lam = sched.get_lam(t)
    return np.sum(coeffs * np.sin(ns * np.pi * lam))


def dlam_sin_coupling(t, sched, *args):
    """Given some variational coefficients and the mode number they are
    associated with, return the lambda-derivative of the sinusoidal
    coupling at a given time
    Parameters:
        t (float):          Time at which to evaluate the coupling
        sched (Schedule):   Schedule object for the lambda schedule
        args (list):        First half of args are integers, respresenting
                            the harmonic number. Second half are floats,
                            represent the coefficients. Need this hacky
                            workaround because QuSpin hashes dynamic
                            parts with arguments of coupling function
    """
    args = np.array(args)
    ns = args[: len(args) // 2]
    coeffs = args[len(args) // 2 :]
    lam = sched.get_lam(t)
    return np.sum(np.pi * ns * coeffs * np.cos(np.pi * ns * lam))


def turn_off_coupling(t, sched):
    """Return $1-\lambda(t)$ which is the coupling for H0
    Parameters:
        t (float):          Time at which to evaluate the coupling
        sched (Schedule):   Schedule object for the lambda schedule
    """
    return 1 - sched.get_lam(t)


def turn_on_coupling(t, sched):
    """Return $\lambda(t)$ which is the coupling for H1
    Parameters:
        t (float):          Time at which to evaluate the coupling
        sched (Schedule):   Schedule object for the lambda schedule
    """
    return sched.get_lam(t)


def sweep_sign(t, sched):
    """Return $2 * \lambda(t) - 1$ which sweeps the sign of the coupling
    from -1 to +1 over the duration of the protocol
    Parameters:
        t (float):          Time at which to evaluate the coupling
        sched (Schedule):   Schedule object for the lambda schedule
    """
    return 2 * sched.get_lam(t) - 1


def off_on_off_sweep(t, sched):
    """Return $1 - |2 * \lambda(t) - 1|$ which sweeps the coupling from
    0 to 1 (halfway) and back to 0 over the duration of the protocol
    Parameters:
        t (float):          Time at which to evaluate the coupling
        sched (Schedule):   Schedule object for the lambda schedule
    """
    return 1 - np.abs(2 * sched.get_lam(t) - 1)


def dlam_off_on_off_sweep(t, sched):
    """Return derivative of $1 - |2 * \lambda(t) - 1|$ which sweeps the,
    coupling from 0 to 1 (halfway) and back to 0 over the duration of the
    protocol, which has a discountinuity at $\lambda = 0.5$
    Parameters:
        t (float):          Time at which to evaluate the coupling
        sched (Schedule):   Schedule object for the lambda schedule
    """
    return 2 - 4 * np.heaviside((sched.get_lam(t) - 0.5), 0.5)
