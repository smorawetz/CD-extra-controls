import numpy as np


def neighbours_1d(N, boundary_conds):
    """Get neighbouring pairs of sites in a 1D chain"""
    if boundary_conds == "periodic":
        return [[i, (i + 1) % N] for i in range(N)]
    elif boundary_conds == "open":
        return [[i, i + 1] for i in range(N - 1)]
    else:
        raise ValueError("{0} is not a valid boundary condition".format(boundary_conds))


def next_neighbours_1d(N, boundary_conds):
    """Get next neighbouring pairs of sites in a 1D chain"""
    if boundary_conds == "periodic":
        return [[i, (i + 2) % N] for i in range(N)]
    elif boundary_conds == "open":
        return [[i, i + 2] for i in range(N - 2)]
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


def neighbours_2d(Nx, Ny, boundary_conds):
    """Get neighbouring pairs of sites on a 2D lattice"""
    Ns = Nx * Ny
    sites = np.arange(Ns)
    r_nbrs = (sites + 1) % Nx + (sites // Nx) * Nx  # right neighbours
    u_nbrs = (sites + Nx) % (Ns)  # up neighbours
    if boundary_conds == "periodic":
        x_nbrs = [[sites[i], r_nbrs[i]] for i in range(Ns)]
        y_nbrs = [[sites[i], u_nbrs[i]] for i in range(Ns)]
        return x_nbrs + y_nbrs
    elif boundary_conds == "open":
        r_inds = (sites + 1) % Nx != 0
        u_inds = (sites + Nx) // Ns != 1
        x_nbrs = [[sites[r_inds][i], r_nbrs[r_inds][i]] for i in range(sum(r_inds))]
        y_nbrs = [[sites[u_inds][i], u_nbrs[u_inds][i]] for i in range(sum(u_inds))]
        return x_nbrs + y_nbrs
    else:
        raise ValueError("{0} is not a valid boundary condition".format(boundary_conds))
