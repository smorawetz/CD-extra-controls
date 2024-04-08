def turn_on(t, sched):
    """Coupling to the term, $\lambda$, which is being turned on"""
    return sched.get_lam(t)


def turn_off(t, sched):
    """Coupling to the term, $1 - \lambda$, which is being turned off"""
    return 1 - sched.get_lam(t)
