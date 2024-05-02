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
