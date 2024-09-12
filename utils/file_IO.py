import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import glob
import h5py
import numpy as np

from utils.file_naming import combine_names


def open_file(file_name, mode="a"):
    return h5py.File("h5data/{0}.h5".format(file_name), mode)


def save_data_agp_coeffs(
    file_name, protocol_name, ctrls_name, tgrid, coeff_grid, lanc_grid=None
):
    """Saves the raw data from the AGP variational optimization into text files
    under the tags file_name, protocol_name, and ctrls_name
    Parameters:
        file_name (str):            name of the HDF5 file
        protocol_name (str):        name of the protocol which indexes subgroup
        ctrls_name (str):           name of the controls scheme which indexes dataset
        tgrid (np.ndarray):         grid of time points
        coeff_grid (np.ndarray):    grid of AGP coefficients
        lanc_grid (np.ndarray):     grid of Lanczos coefficients, if needed
    """
    save_dirname = "{0}/data_dump".format(os.environ["CD_CODE_DIR"])
    full_info_fname = combine_names(file_name, protocol_name, ctrls_name)
    ctrls_free_fname = combine_names(file_name, protocol_name)
    tgrid_path = "{0}/agp_coeffs/{1}_tgrid.txt".format(save_dirname, ctrls_free_fname)
    agp_coeffs_path = "{0}/agp_coeffs/{1}".format(save_dirname, full_info_fname)

    np.savetxt("{0}_tgrid.txt".format(tgrid_path), tgrid)
    np.savetxt("{0}_coeffs_grid.txt".format(agp_coeffs_path), coeff_grid)
    if lanc_grid is not None:
        np.savetxt("{0}_lanc_grid.txt".format(agp_coeffs_path), lanc_grid)
    return None


def load_raw_data_agp_coeffs(file_name, protocol_name, ctrls_name):
    """Loads the raw data from the AGP variational optimization into arrays
    Parameters:
        file_name (str):            name of the HDF5 file
        protocol_name (str):        name of the protocol which indexes subgroup
        ctrls_name (str):           name of the controls scheme which indexes dataset
    """
    save_dirname = "{0}/data_dump".format(os.environ["CD_CODE_DIR"])
    full_info_fname = combine_names(file_name, protocol_name, ctrls_name)
    ctrls_free_fname = combine_names(file_name, protocol_name)
    tgrid_path = "{0}/agp_coeffs/{1}_tgrid.txt".format(save_dirname, ctrls_free_fname)
    agp_coeffs_path = "{0}/agp_coeffs/{1}".format(save_dirname, full_info_fname)
    tgrid = np.loadtxt("{0}_tgrid.txt".format(tgrid_path))
    coeff_grid = np.loadtxt("{0}_coeffs_grid.txt".format(agp_coeffs_path))
    if "kry" in protocol_name:
        lanc_grid = np.loadtxt("{0}_lanc_grid.txt".format(agp_coeffs_path))
    else:
        lanc_grid = None
    return tgrid, coeff_grid, lanc_grid


def merge_data_agp_coeffs(
    file_name, protocol_name, ctrls_name, tgrid, coeff_grid, lanc_grid=None
):
    """Merges the raw data from the AGP variational optimization into an HDF5 file
    file_name, in the agp_coeffs group under the protocol_name subgroup, and dataset
    files indexed by the controls scheme controls_name
    Parameters:
        file_name (str):            name of the HDF5 file
        protocol_name (str):        name of the protocol which indexes subgroup
        ctrls_name (str):           name of the controls scheme which indexes dataset
        tgrid (np.ndarray):         grid of time points
        coeff_grid (np.ndarray):    grid of AGP coefficients
        lanc_grid (np.ndarray):     grid of Lanczos coefficients, if needed
    """
    f = open_file(file_name)
    protocol_grp = f.require_group("agp_coeffs/{0}".format(protocol_name))
    tgrid_dataset = protocol_grp.require_dataset("tgrid", tgrid.shape, tgrid.dtype)
    tgrid_dataset[:] = tgrid
    coeff_dataset = protocol_grp.require_dataset(
        "{0}_coeffs_grid".format(ctrls_name), coeff_grid.shape, coeff_grid.dtype
    )
    coeff_dataset[:] = coeff_grid
    if lanc_grid is not None:
        lanc_dataset = protocol_grp.require_dataset(
            "{0}_lanc_grid".format(ctrls_name), lanc_grid.shape, lanc_grid.dtype
        )
        lanc_dataset[:] = lanc_grid
    f.close()
    return None


def load_data_agp_coeffs(file_name, protocol_name, ctrls_name):
    """Loads the data from the AGP variational optimization from an HDF5 file
    file_name, in the agp_coeffs group under the protocol_name subgroup, and dataset
    files indexed by the controls scheme controls_name
    Parameters:
        file_name (str):            name of the HDF5 file
        protocol_name (str):        name of the protocol which indexes subgroup
        ctrls_name (str):           name of the controls scheme which indexes dataset
    """
    f = open_file(file_name, mode="r")
    protocol_grp = f.require_group("agp_coeffs/{0}".format(protocol_name))
    tgrid = protocol_grp["tgrid"][:]
    coeffs_grid = protocol_grp["{0}_coeffs_grid".format(ctrls_name)][:].reshape(
        len(tgrid), -1
    )
    if "kry" in protocol_name:
        lanc_grid = protocol_grp["{0}_lanc_grid".format(ctrls_name)][:]
    else:
        lanc_grid = None
    return tgrid, coeffs_grid, lanc_grid


def save_data_optimization_fids(file_name, protocol_name, ctrls_name, coeffs, fid):
    """Saves the raw data from the controls coefficients optimization into text files
    under the tags file_name, protocol_name, ctrls_name
    Parameters:
        file_name (str):            name of the HDF5 file
        protocol_name (str):        name of the protocol which indexes subgroup
        ctrls_name (str):           name of the controls scheme which indexes dataset
        coeffs (np.ndarray):        coefficients of the control scheme
    """
    save_dirname = "{0}/data_dump".format(os.environ["CD_CODE_DIR"])
    full_info_fname = combine_names(file_name, protocol_name, ctrls_name)
    optim_list_fname = "{0}/ctrls_optim/{1}.txt".format(save_dirname, full_info_fname)
    with open(optim_list_fname, "a") as data_file:
        np.savetxt(data_file, np.expand_dims(np.array([*coeffs, fid]), axis=0))
    return None


def load_raw_data_optimization_fids(file_name, protocol_name, ctrls_name):
    """Loads the raw data from the controls optimization into arrays
    Parameters:
        file_name (str):            name of the HDF5 file
        protocol_name (str):        name of the protocol which indexes subgroup
        ctrls_name (str):           name of the controls scheme which indexes dataset
    """
    pass
    save_dirname = "{0}/data_dump".format(os.environ["CD_CODE_DIR"])
    full_info_fname = combine_names(file_name, protocol_name, ctrls_name)
    optim_list_fname = "{0}/ctrls_optim/{1}.txt".format(save_dirname, full_info_fname)
    data_file = np.loadtxt(optim_list_fname)
    coeffs = data_file[:, :-1]
    fids = data_file[:, -1]
    return coeffs, fids


def merge_data_optimization_fids(file_name, protocol_name, ctrls_name, coeffs, fids):
    """Merges the raw data from the extra controls optimization into an HDF5 file
    file_name, in the optimization_fids group under the protocol_name subgroup, and dataset
    files indexed by the controls scheme controls_name
    Parameters:
        file_name (str):            name of the HDF5 file
        protocol_name (str):        name of the protocol which indexes subgroup
        ctrls_name (str):           name of the controls scheme which indexes dataset
        coeffs (np.ndarray):        array of with different choices of coefficients in
                                    the rows
        fids (np.ndarray):          fidelity of final state for control scheme in the above
    """
    f = open_file(file_name)
    protocol_grp = f.require_group("optimization_fids/{0}".format(protocol_name))
    data = np.hstack([coeffs, fids.reshape(-1, 1)])
    optimization_dataset = protocol_grp.require_dataset(
        "{0}_optim_data".format(ctrls_name), data.shape, data.dtype
    )
    optimization_dataset[:] = data
    f.close()
    return None


def load_data_optimization_fids(file_name, protocol_name, ctrls_name):
    f = open_file(file_name)
    protocol_grp = f.require_group("optimization_fids/{0}".format(protocol_name))
    optimization_dataset = protocol_grp["{0}_optim_data".format(ctrls_name)]
    coeffs_data = optimization_dataset[:, :-1]
    fids_data = optimization_dataset[:, -1]
    return coeffs_data, fids_data


def save_data_evolved_wfs(
    file_name, protocol_name, ctrls_name, final_wf, tgrid=None, full_wf=None
):
    """Saves the raw data from the evolving the intial wavefunction under some control
    scheme with an approximate AGP under the tags file_name, protocol_name, ctrls_name
    Parameters:
        file_name (str):            name of the HDF5 file
        protocol_name (str):        name of the protocol which indexes subgroup
        ctrls_name (str):           name of the controls scheme which indexes dataset
        final_wf (np.ndarray):      final wavefunction after evolution with given control
        tgrid (np.ndarray):         grid of time points
        full_wf (np.ndarray):       wavefunction at all points at a given time,
                                    associated with points on tgrid
    """
    save_dirname = "{0}/data_dump".format(os.environ["CD_CODE_DIR"])
    full_info_fname = combine_names(file_name, protocol_name, ctrls_name)
    tgrid_fname = combine_names(file_name, protocol_name)
    evolved_wfs_path = "{0}/evolved_wfs/{1}".format(save_dirname, full_info_fname)
    tgrid_path = "{0}/evolved_wfs/{1}".format(save_dirname, tgrid_fname)
    np.savetxt("{0}_final_wf.txt".format(evolved_wfs_path), final_wf)
    if tgrid is not None:
        np.savetxt("{0}_tgrid.txt".format(tgrid_path), tgrid)
        np.savetxt("{0}_full_evolution_wf.txt".format(evolved_wfs_path), full_wf)
    return None


def load_raw_data_evolved_wfs(file_name, protocol_name, ctrls_name):
    """Loads the raw data from the evolving the intial wavefunction under some control
    scheme with an approximate AGP under the tags file_name, protocol_name, ctrls_name
    Parameters:
        file_name (str):            name of the HDF5 file
        protocol_name (str):        name of the protocol which indexes subgroup
        ctrls_name (str):           name of the controls scheme which indexes dataset
    """
    save_dirname = "{0}/data_dump".format(os.environ["CD_CODE_DIR"])
    full_info_fname = combine_names(file_name, protocol_name, ctrls_name)
    tgrid_fname = combine_names(file_name, protocol_name)
    evolved_wfs_path = "{0}/evolved_wfs/{1}".format(save_dirname, full_info_fname)
    tgrid_path = "{0}/evolved_wfs/{1}".format(save_dirname, tgrid_fname)
    final_wf = np.loadtxt(
        "{0}_final_wf.txt".format(evolved_wfs_path), dtype=np.complex128
    )
    if os.path.isfile("{0}_tgrid.txt".format(tgrid_path)):
        tgrid = np.loadtxt("{0}_tgrid.txt".format(tgrid_path))
    else:
        tgrid = None
    if os.path.isfile("{0}_full_evolution_wf.txt".format(evolved_wfs_path)):
        full_wf = np.loadtxt(
            "{0}_full_evolution_wf.txt".format(evolved_wfs_path), dtype=np.complex128
        )
    else:
        full_wf = None
    return final_wf, tgrid, full_wf


def merge_data_evolved_wfs(
    file_name, protocol_name, ctrls_name, final_wf, tgrid=None, full_wf=None
):
    """Merges the raw data from the extra controls optimization into an HDF5 file
    file_name, in the optimization_fids group under the protocol_name subgroup, and dataset
    files indexed by the controls scheme controls_name
    Parameters:
        file_name (str):            name of the HDF5 file
        protocol_name (str):        name of the protocol which indexes subgroup
        ctrls_name (str):           name of the controls scheme which indexes dataset
        final_wf (np.ndarray):      final wavefunction after evolution with given control
        tgrid (np.ndarray):         grid of time points
        full_wf (np.ndarray):       wavefunction at all points at a given time,
                                    associated with points on tgrid
    """
    f = open_file(file_name)
    protocol_grp = f.require_group("evolved_wfs/{0}".format(protocol_name))
    final_wf_dataset = protocol_grp.require_dataset(
        "{0}_final_wf".format(ctrls_name), final_wf.shape, final_wf.dtype
    )
    final_wf_dataset[:] = final_wf
    if tgrid is not None:
        tgrid_dataset = protocol_grp.require_dataset("tgrid", tgrid.shape, tgrid.dtype)
        tgrid_dataset[:] = tgrid
    if full_wf is not None:
        full_wf_dataset = protocol_grp.require_dataset(
            "{0}_full_wf".format(ctrls_name), full_wf.shape, full_wf.dtype
        )
        full_wf_dataset[:] = full_wf
    f.close()
    return None


def load_data_evolved_wfs(file_name, protocol_name, ctrls_name, get_full_wf=False):
    """Loads the data from the evolved wavefunctions from an HDF5 file
    file_name, in the agp_coeffs group under the protocol_name subgroup, and dataset
    files indexed by the controls scheme controls_name
    Parameters:
        file_name (str):            name of the HDF5 file
        protocol_name (str):        name of the protocol which indexes subgroup
        ctrls_name (str):           name of the controls scheme which indexes dataset
        get_full_wf (bool):         if True, returns the full wavefunction at all points
    """
    f = open_file(file_name, mode="r")
    protocol_grp = f.require_group("evolved_wfs/{0}".format(protocol_name))
    final_wf_dataset = protocol_grp["{0}_final_wf".format(ctrls_name)]
    final_wf = final_wf_dataset[:]
    if get_full_wf:
        tgrid_dataset = protocol_grp["tgrid"]
        tgrid = tgrid_dataset[:]
        full_wf_dataset = protocol_grp["{0}_full_wf".format(ctrls_name)]
        full_wf = full_wf_dataset[:]
    else:
        tgrid = None
        full_wf = None
    return final_wf, tgrid, full_wf


def save_data_evolved_wfs_blocks(
    file_name,
    protocol_name,
    ctrls_name,
    final_wf_blocks,
):
    """Saves the raw data from the evolving the intial wavefunction under some control
    scheme with an approximate AGP under the tags file_name, protocol_name, ctrls_name
    Parameters:
        file_name (str):                name of the HDF5 file
        protocol_name (str):            name of the protocol which indexes subgroup
        ctrls_name (str):               name of the controls scheme which indexes dataset
        final_wf_blocks (np.ndarray):   array with first column as k blocks, later
                                        columns are components of wavefunction
    """
    save_dirname = "{0}/data_dump".format(os.environ["CD_CODE_DIR"])
    full_info_fname = combine_names(file_name, protocol_name, ctrls_name)
    evolved_wfs_path = "{0}/evolved_wfs/{1}".format(save_dirname, full_info_fname)
    np.savetxt("{0}_final_wf_blocks.txt".format(evolved_wfs_path), final_wf_blocks)
    return None


def load_raw_data_evolved_wfs_blocks(file_name, protocol_name, ctrls_name):
    """Loads the raw data from the evolving the intial wavefunction under some control
    scheme with an approximate AGP under the tags file_name, protocol_name, ctrls_name
    Parameters:
        file_name (str):            name of the HDF5 file
        protocol_name (str):        name of the protocol which indexes subgroup
        ctrls_name (str):           name of the controls scheme which indexes dataset
    """
    save_dirname = "{0}/data_dump".format(os.environ["CD_CODE_DIR"])
    full_info_fname = combine_names(file_name, protocol_name, ctrls_name)
    evolved_wfs_path = "{0}/evolved_wfs/{1}".format(save_dirname, full_info_fname)
    final_wf_blocks = np.loadtxt(
        "{0}_final_wf_blocks.txt".format(evolved_wfs_path), dtype=np.complex128
    )
    tgrid = None
    full_wf = None
    return final_wf_blocks, tgrid, full_wf


def merge_data_evolved_wfs_blocks(
    file_name, protocol_name, ctrls_name, final_wf_blocks, tgrid=None, full_wf=None
):
    """Merges the raw data from the extra controls optimization into an HDF5 file
    file_name, in the optimization_fids group under the protocol_name subgroup, and dataset
    files indexed by the controls scheme controls_name
    Parameters:
        file_name (str):                name of the HDF5 file
        protocol_name (str):            name of the protocol which indexes subgroup
        ctrls_name (str):               name of the controls scheme which indexes dataset
        final_wf_blocks (np.ndarray):   final wavefunction after evolution with given control
    """
    f = open_file(file_name)
    protocol_grp = f.require_group("evolved_wfs/{0}".format(protocol_name))
    final_wf_dataset = protocol_grp.require_dataset(
        "{0}_final_wf_blocks".format(ctrls_name),
        final_wf_blocks.shape,
        final_wf_blocks.dtype,
    )
    final_wf_dataset[:] = final_wf_blocks
    f.close()
    return None


def load_data_evolved_wfs_blocks(
    file_name, protocol_name, ctrls_name, get_full_wf=False
):
    """Loads the data from the evolved wavefunctions from an HDF5 file
    file_name, in the agp_coeffs group under the protocol_name subgroup, and dataset
    files indexed by the controls scheme controls_name
    Parameters:
        file_name (str):            name of the HDF5 file
        protocol_name (str):        name of the protocol which indexes subgroup
        ctrls_name (str):           name of the controls scheme which indexes dataset
    """
    f = open_file(file_name, mode="r")
    protocol_grp = f.require_group("evolved_wfs/{0}".format(protocol_name))
    final_wf_dataset = protocol_grp["{0}_final_wf_blocks".format(ctrls_name)]
    final_wf_blocks = final_wf_dataset[:]
    tgrid = None
    full_wf = None
    return final_wf_blocks, tgrid, full_wf


def delete_data_dump_files(
    Ns,
    model_name,
    H_params,
    symmetries,
    ctrls,
    ctrls_couplings,
    ctrl_args,
    agp_order,
    AGPtype,
    norm_type,
    grid_size,
    sched,
):
    h5fname = make_file_name(Ns, model_name, H_params, symmetries, ctrls)
    f = open_file(file_name)
    if "agp_coeffs" not in f or "ctrls_optim" not in f or "evolved_wfs" not in f.keys():
        print("some data is missing in the HDF5 file!!")
        return None

    file_name, protocol_name, ctrls_name = make_data_dump_name(
        Ns,
        model_name,
        H_params,
        sched,
        symmetries,
        ctrls,
        ctrls_couplings,
        ctrls_args,
        agp_order,
        AGPtype,
        norm_type,
        grid_size,
    )
    save_dirname = "{0}/data_dump".format(os.environ["CD_CODE_DIR"])
    agp_coeffs_path = "{0}/agp_coeffs/{1}".format(save_dirname, data_dump_fname)
    ctrls_optim_path = "{0}/ctrls_optim/{1}".format(save_dirname, data_dump_fname)
    evolved_wfs_path = "{0}/evolved_wfs/{1}".format(save_dirname, data_dump_fname)
    for txtf in glob.glob("{0}*".format(agp_coeffs_path)):
        os.remove(txtf)
    for txtf in glob.glob("{0}*".format(ctrls_optim_path)):
        os.remove(txtf)
    for txtf in glob.glob("{0}*".format(evolved_wfs_path)):
        os.remove(txtf)
    f.close()
    return None
