def neighbours_1d(N, boundary_conds):
    """Get neighbouring pairs of sites in a 1D chain"""
    if boundary_conds == "periodic":
        return [[i, (i + 1) % N] for i in range(N)]
    elif boundary_conds == "open":
        return [[i, i + 1] for i in range(N - 1)]
    else:
        raise ValueError("{0} is not a valid boundary condition".format(boundary_conds))


def triplets_1d(N, boundary_conds):
    """Get all triplets of neighbouring spins (3 in a row) in a 1D chain"""
    if boundary_conds == "periodic":
        return [[i, (i + 1) % N, (i + 2) % N] for i in range(N)]
    elif boundary_conds == "open":
        return [[i, i + 1, i + 2] for i in range(N - 2)]
    else:
        raise ValueError("{0} is not a valid boundary condition".format(boundary_conds))


def all_connections_1d(N, boundary_conds):
    """Find interactions between all pairs of sites in a 1D chain, and the
    distance between them (works for PBC and OBC)"""
    pairs = list(itertools.product(range(N), range(N)))
    for i in range(N):  # remove self-interaction
        pairs.remove((i, i))
    # if PBC, set pair distance to be smaller if can reduce by wrapping around
    if boundary_conds == "periodic":
        pair_distances = list(
            map(
                lambda x: min(abs(x[1] - x[0]), Nspins - abs(x[1] - x[0])),
                pairs,
            )
        )
    elif boundary_conditions == "open":
        pair_distances = list(map(lambda x: abs(x[1] - x[0]), pairs))
    else:
        raise ValueError("{0} is not a valid boundary condition".format(boundary_conds))
    return pairs, pair_distances
