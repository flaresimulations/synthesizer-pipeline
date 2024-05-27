import os
import time
import gc
import argparse
import fnmatch
from functools import partial
import numpy as np
from numpy.typing import NDArray
from collections import namedtuple
from typing import List, Optional, Any, Dict
import h5py

from mpi4py import MPI
from astropy.cosmology import LambdaCDM
from schwimmbad import MultiPool
from unyt import Angstrom, erg, s

from synthesizer.load_data.load_eagle import read_array, get_age
from synthesizer.grid import Grid
from synthesizer.filters import FilterCollection
from synthesizer.kernel_functions import Kernel
from synthesizer.sed import Sed, combine_list_of_seds
from synthesizer.load_data.load_eagle import load_EAGLE_shm
from synthesizer.load_data.utils import get_len

from utilities import (
    save_dummy_file, get_spectra, get_lines
)

# define EAGLE cosmology
cosmo = LambdaCDM(Om0=0.307, Ode0=0.693, H0=67.77, Ob0=0.04825)


def create_shm_nparray(
        data_array: NDArray[Any],
        mmap_dest: str,
        NP_DATA_TYPE: str = 'float32'
        ) -> None:
    """
    Arguments:
        data_array (array)
            data array to create memmap
        mmap_dest (str)
            location and name of memmap
        NP_DATA_TYPE (str)
            data type of the array
    """

    array_shape = data_array.shape

    dst = np.memmap(
        mmap_dest, dtype=NP_DATA_TYPE, mode='w+', shape=array_shape
        )
    np.copyto(dst, data_array)


def np_shm_read(
        name: str,
        array_shape: tuple,
        dtype: str = 'float32'
        ) -> NDArray[Any]:
    """
    Arguments:
        name (str)
            destination and name of memmap array
        array_shape (tuple)
            shape of the memmap array
        dtype (str)
            data type of the memmap array
    """

    tmp = np.memmap(name, dtype=dtype, mode='r', shape=array_shape)
    return tmp


def create_eagle_shm(
        args: namedtuple,
        nthreads: Optional[int] = None
        ) -> None:
    """
    Arguments:
        args (NamedTuple):
            parser arguments passed on to this job
        nthreads (int)
            number of threads to use
    """

    zed = float(args.tag[5:].replace("p", "."))
    
    if nthreads is None:
        nthreads = args.nthreads

    # Get required star particle properties
    tmp = read_array(
        "PARTDATA",
        args.eagle_file,
        args.tag,
        "/PartType4/SubGroupNumber",
        numThreads=nthreads
    ).astype(np.int32)
    s_len = len(tmp)
    create_shm_nparray(tmp, F'{args.shm_prefix}s_sgrpno{args.shm_suffix}', NP_DATA_TYPE='int32')
    
    tmp = read_array(
        "PARTDATA",
        args.eagle_file,
        args.tag,
        "/PartType4/GroupNumber",
        numThreads=nthreads
    ).astype(np.int32)
    create_shm_nparray(tmp, F'{args.shm_prefix}s_grpno{args.shm_suffix}', NP_DATA_TYPE='int32')
    
    tmp = read_array(
        "PARTDATA",
        args.eagle_file,
        args.tag,
        "/PartType4/Coordinates",
        noH=True,
        physicalUnits=True,
        numThreads=nthreads
    ).astype(np.float32)  # physical Mpc
    create_shm_nparray(tmp, F'{args.shm_prefix}s_coords{args.shm_suffix}')

    tmp = (
        read_array(
            "PARTDATA",
            args.eagle_file,
            args.tag,
            "/PartType4/InitialMass",
            noH=True,
            physicalUnits=True,
            numThreads=nthreads
        ).astype(np.float32)
        * 1e10
    )  #  Msun
    create_shm_nparray(tmp, F'{args.shm_prefix}s_imasses{args.shm_suffix}')

    tmp = (
        read_array(
            "PARTDATA",
            args.eagle_file,
            args.tag,
            "/PartType4/Mass",
            noH=True,
            physicalUnits=True,
            numThreads=nthreads
        ).astype(np.float32)
        * 1e10
    )  #  Msun
    create_shm_nparray(tmp, F'{args.shm_prefix}s_masses{args.shm_suffix}')

    tmp = read_array(
        "PARTDATA",
        args.eagle_file,
        args.tag,
        "/PartType4/StellarFormationTime",
        numThreads=nthreads
    ).astype(np.float32)
    tmp = get_age(tmp, zed, numThreads=args.nthreads)  # Gyr
    create_shm_nparray(tmp, F'{args.shm_prefix}s_ages{args.shm_suffix}')

    tmp = read_array(
        "PARTDATA",
        args.eagle_file,
        args.tag,
        "/PartType4/SmoothedMetallicity",
        numThreads=nthreads
    ).astype(np.float32)
    create_shm_nparray(tmp, F'{args.shm_prefix}s_Zsmooth{args.shm_suffix}')

    tmp = read_array(
        "PARTDATA",
        args.eagle_file,
        args.tag,
        "/PartType4/ElementAbundance/Oxygen",
        numThreads=nthreads
    ).astype(np.float32)
    create_shm_nparray(tmp, F'{args.shm_prefix}s_oxygen{args.shm_suffix}')

    tmp = read_array(
        "PARTDATA",
        args.eagle_file,
        args.tag,
        "/PartType4/ElementAbundance/Hydrogen",
        numThreads=nthreads
    ).astype(np.float32)
    create_shm_nparray(tmp, F'{args.shm_prefix}s_hydrogen{args.shm_suffix}')

    # Get gas particle properties
    tmp = read_array(
        "PARTDATA",
        args.eagle_file,
        args.tag,
        "/PartType0/SubGroupNumber",
        numThreads=nthreads
    ).astype(np.int32)
    g_len = len(tmp)
    create_shm_nparray(tmp, F'{args.shm_prefix}g_sgrpno{args.shm_suffix}', NP_DATA_TYPE='int32')

    tmp = read_array(
        "PARTDATA",
        args.eagle_file,
        args.tag,
        "/PartType0/GroupNumber",
        numThreads=nthreads
    ).astype(np.int32)
    create_shm_nparray(tmp, F'{args.shm_prefix}g_grpno{args.shm_suffix}', NP_DATA_TYPE='int32')

    tmp = read_array(
        "PARTDATA",
        args.eagle_file,
        args.tag,
        "/PartType0/Coordinates",
        noH=True,
        physicalUnits=True,
        numThreads=nthreads
    ).astype(np.float32)  # physical Mpc
    create_shm_nparray(tmp, F'{args.shm_prefix}g_coords{args.shm_suffix}')

    tmp = (
        read_array(
            "PARTDATA",
            args.eagle_file,
            args.tag,
            "/PartType0/Mass",
            noH=True,
            physicalUnits=True,
            numThreads=nthreads
        )
        * 1e10
    ).astype(np.float32)  # Msun
    create_shm_nparray(tmp, F'{args.shm_prefix}g_masses{args.shm_suffix}')

    tmp = read_array(
        "PARTDATA",
        args.eagle_file,
        args.tag,
        "/PartType0/StarFormationRate",
        numThreads=nthreads
    ).astype(np.float32)  # Msol / yr
    create_shm_nparray(tmp, F'{args.shm_prefix}g_sfr{args.shm_suffix}')

    tmp = read_array(
        "PARTDATA",
        args.eagle_file,
        args.tag,
        "/PartType0/SmoothedMetallicity",
        numThreads=nthreads
    ).astype(np.float32)
    create_shm_nparray(tmp, F'{args.shm_prefix}g_Zsmooth{args.shm_suffix}')

    tmp = read_array(
        "PARTDATA",
        args.eagle_file,
        args.tag,
        "/PartType0/SmoothingLength",
        noH=True,
        physicalUnits=True,
        numThreads=nthreads
    ).astype(np.float32)  # physical Mpc
    create_shm_nparray(tmp, F'{args.shm_prefix}g_hsml{args.shm_suffix}')

    np.savez(
        F'{args.shm_prefix}lengths{args.shm_suffix}', s_len=s_len, g_len=g_len
        )


def save_photometry_hdf5(
        ii: int,
        output_file: str,
        fluxes: Dict,
        luminosities: Dict,
        lines: List[Dict],
        spec_keys: List,
        line_list: List,
        snaps: NDArray[np.int32],
        begin: NDArray[np.int32],
        end: NDArray[np.int32]
        ):

    snapnum = snaps[ii]
    # # Save spectra, fluxes and luminosities
    with h5py.File(F'{output_file}.{snapnum}.hdf5', 'w') as hf:

        # Loop through different spectra / dust models
        for key in spec_keys:
            # grp = hf.require_group('SED')
            # dset = grp.create_dataset(f'{str(key)}', data=specs[key].lnu)
            # dset.attrs['Units'] = str(specs[key].lnu.units)
            # # Include wavelength array corresponding to SEDs
            # if (key == list(dat[0].keys())[0]) * (chunk == 0):
            #     lam = grp.create_dataset('Wavelength', data=specs[key].lam)
            #     lam.attrs['Units'] = str(specs[key].lam.units)

            grp = hf.require_group('Fluxes')
            # Create separate groups for different instruments
            for f in fluxes[key].filters:
                dset = grp.create_dataset(
                    f'{str(key)}/{f.filter_code}',
                    data=fluxes[key][f.filter_code][begin[ii]:end[ii]],
                    dtype=np.float32
                )

                dset.attrs['Units'] = str(fluxes[key].photometry.units)

            grp = hf.require_group('Luminosities')
            # Create separate groups for different instruments
            for f in luminosities[key].filters:
                dset = grp.create_dataset(
                    f'{str(key)}/{f.filter_code}',
                    data=luminosities[key][f.filter_code][begin[ii]:end[ii]],
                    dtype=np.float32
                )

                dset.attrs['Units'] = str(luminosities[key].photometry.units)
        
            if key != 'stellar':
                grp = hf.require_group('Lines')
                # Create separate groups for different nebular lines
                for jj, f in enumerate(line_list):
                    dset = grp.create_dataset(
                    f'{str(key)}/{f}/Luminosities',
                    data=lines[0][key][:,jj][begin[ii]:end[ii]],
                    dtype=np.float32
                    )

                    dset.attrs['Units'] = str(lines[0][key].units)

                    dset = grp.create_dataset(
                    f'{str(key)}/{f}/EWs',
                    data=lines[1][key][:,jj][begin[ii]:end[ii]],
                    dtype=np.float32
                    )

                    dset.attrs['Units'] = str(lines[1][key].units)



def generate_and_save_photometry(
        chunks: NDArray[np.int32],
        args: namedtuple,
        grid: Grid,
        fc: List[FilterCollection],
        spec_keys: List,
        line_list: List,
        s_len: int,
        g_len: int,
        kern: Optional[Kernel] = None,
        tot_chunks: int = 1535
        ) -> None:
    """
    Read in the eagle galaxy data for the chunk subfind file and
    write out the required photometry information in spec_keys
    Arguments:
        chunk (array)
            file numbers to process
        args (NamedSpace):
            parser arguments passed on to this job
        grid (Grid object)
            SPS grid object to generate the spectra
        fc (FilterCollection object)
            filters to generate the photometry
        spec_keys (list)
            list of galaxy spectra type to generate
        s_len (int):
            total number of star particles
        g_len (int):
            total number of gas particles
        kern (Kernel object, optional)
            SPH kernel to use for line-of-sight dust attenuation
        tot_chunks (int)
            total number of files to process
    """
    fc_flux, fc_lum = fc

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    _f = partial(
        load_EAGLE_shm,
        fileloc=args.eagle_file,
        tag=args.tag,
        s_len=s_len,
        g_len=g_len,
        args=args,
        numThreads=1,
        tot_chunks=tot_chunks)

    start = time.time()
    with MultiPool(processes=4*args.nthreads) as pool:
        gals_chunks = np.array(pool.map(_f, chunks), dtype=object)
    end = time.time()

    if gals_chunks.size == 0:
        for ii in chunks:
            save_dummy_file(
                F"{args.output}.{ii}.hdf5",
                filters=fc,
                line_list=line_list,
                keys=spec_keys
            )
        return None
        
    gc.collect()

    empty_chunks = np.where([ii==[] for ii in gals_chunks])[0]
    # if there are empty snaps
    if len(empty_chunks) > 0:
        # delete empty snaps
        gals_chunks = np.delete(gals_chunks, empty_chunks)
        # create dummy files for empty chunks
        print(F"No galaxies in these chunks: {chunks[empty_chunks]}")
        for ii in chunks[empty_chunks]:
            save_dummy_file(
                F"{args.output}.{ii}.hdf5",
                filters=fc,
                line_list=line_list,
                keys=spec_keys
            )

    # Calculate number of galaxies in each remaining chunk
    gal_lengths = np.array([len(ii) for ii in gals_chunks])
    gals_chunks = np.concatenate(gals_chunks)
    nonempty_snaps = np.delete(chunks, empty_chunks)

    start = time.time()
    _f = partial(get_spectra, grid=grid, spec_keys=spec_keys, kern=kern)
    print(F"Generating spectrum, number of galaxies in rank {rank}: ", len(gals_chunks))

    with MultiPool(processes=args.nthreads) as pool:
        dat = np.array(pool.map(_f, gals_chunks), dtype=object)

    gals_chunks = np.array(dat[:,1], dtype=object)
    dat = list(dat[:,0])
    gc.collect()
    end = time.time()
    print(f"Spectra generation on rank {rank} took: {end - start:.2f}")

    # # galaxies that don't have stellar particles
    mask = np.array(dat) == {}
    if np.sum(gal_lengths) == np.sum(mask):
        print(F'No stars in galaxies. Rank {rank} saving dummy file.')
        for ii, jj in enumerate(nonempty_snaps):
            save_dummy_file(
                F"{args.output}.{jj}.hdf5",
                filters=fc,
                line_list=line_list,
                keys=spec_keys,
                gal_num = gal_lengths[ii]
            )
        return None

    # Remove galaxies with zero stars
    gals_chunks = gals_chunks[~mask]
    # Assign null SEDs to galaxies with no stars
    ok = np.where(mask)[0]
    null_sed = Sed(lam=dat[np.where(~mask)[0][0]]['stellar'].wavelength)
    null_dict = {ii: null_sed for ii in spec_keys}
    dat = np.array(dat, dtype=object)
    dat[ok] = null_dict

    specs: Dict = {}
    for key in spec_keys:
        specs[key] = combine_list_of_seds([_dat[key] for _dat in dat])

    # Calculate photometry (observer frame fluxes and luminosities)
    fluxes: Dict = {}
    luminosities: Dict = {}

    start = time.time()
    for key in spec_keys:
        specs[key].get_fnu(cosmo=cosmo, z=gals_chunks[0].redshift)
        fluxes[key] = specs[key].get_photo_fluxes(fc_flux, verbose=False)
        luminosities[key] = specs[key].get_photo_luminosities(fc_lum, verbose=False)

    del specs, dat
    gc.collect()
    end = time.time()
    print(f"Photometry calculation on rank {rank} took: {end - start:.2f}")

    # Calculate line luminosity and EWs
    lines_lum: Dict = {}
    lines_ew: Dict = {}

    line_keys = spec_keys.copy()
    if 'stellar' in line_keys:
        line_keys.remove('stellar')

    start = time.time()
    _f = partial(
        get_lines, grid=grid,
        line_keys=line_keys,
        line_list=line_list,
        kern=kern
        )
    with MultiPool(processes=args.nthreads) as pool:
        dat = pool.map(_f, gals_chunks)

    # make temporary array to store the line outputs
    # of zero star galaxies
    tmp_lum = np.zeros((len(mask), len(line_list))) * erg/s
    tmp_ew = np.zeros((len(mask), len(line_list))) * Angstrom
    ok = np.where(~mask)[0]
    for ii, key in enumerate(line_keys):
        for jj, _line in enumerate(line_list):
            tmp_lum[:,jj][ok] = np.array([np.sum(kk[key][_line]._luminosity) for kk in dat])
            tmp_ew[:,jj][ok] = np.array([np.sum(kk[key][_line]._equivalent_width) for kk in dat])

        lines_lum[key] = tmp_lum
        lines_ew[key] = tmp_ew

    end = time.time()
    print(f"Line calculation on rank {rank} took: {end - start:.2f}")

    begin, end = get_len(gal_lengths)

    _f = partial(
        save_photometry_hdf5, output_file=args.output,
        fluxes=fluxes, luminosities=luminosities,
        lines=[lines_lum, lines_ew], spec_keys=spec_keys,
        line_list=line_list, snaps=nonempty_snaps,
        begin=begin, end=end
        )

    with MultiPool(processes=args.nthreads) as pool:
        pool.map(_f, np.arange(len(nonempty_snaps)))

    print(F"Finished writing out files {chunks[0]}::{size}:{chunks[-1]} on rank {rank}")


if __name__ == "__main__":

    # Setting up MPI if invoked
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(
        description="Run the synthesizer EAGLE pipeline"
        )

    parser.add_argument(
        "-tag",
        type=str,
        help="EAGLE snapshot tag",
    )

    parser.add_argument(
        "-volume",
        type=str,
        required=False,
        help="EAGLE volume name",
        default='REF_100'
    )

    parser.add_argument(
        "-eagle-file",
        type=str,
        required=False,
        help="EAGLE file",
        default='/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data'
    )

    parser.add_argument(
        "-grid-name",
        type=str,
        required=False,
        help="Synthesizer grid file",
        default="bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps"
    )

    parser.add_argument(
        "-grid-directory",
        type=str,
        required=False,
        help="Synthesizer grid directory",
        default="../../synthesizer_data/grids/"
    )

    parser.add_argument(
        "-output",
        type=str,
        required=False,
        help="Output file",
        default="./data/eagle_photometry"
    )

    parser.add_argument(
        "-total-tasks",
        type=int,
        required=False,
        help="Number number of workers used",
        default=1
    )

    parser.add_argument(
        "-total-nodes",
        type=int,
        required=False,
        help="Number of nodes",
        default=1
    )

    parser.add_argument(
        "-cnodenum",
        type=int,
        required=False,
        help="Current node number",
        default=0
    )

    parser.add_argument(
        "-nthreads",
        type=int,
        required=False,
        help="Number of threads",
        default=2
    )

    parser.add_argument(
        "-shm-prefix",
        type=str,
        required=False,
        help="Prefix for the shm names",
        default=''
    )

    parser.add_argument(
        "-shm-suffix",
        type=str,
        required=False,
        help="Suffix for the shm names",
        default=''
    )

    parser.add_argument(
        "-node-name",
        type=str,
        required=False,
        help="Name of node",
        default=''
    )

    parser.add_argument(
        "-n-segments",
        type=int,
        required=False,
        help="Number of segments to run",
        default=''
    )

    args = parser.parse_args()
    output_file = F"{args.output}_{args.volume}_{args.tag}"

    spec_keys: List = ['stellar', 'intrinsic', 'los']
    line_list: List = ['H 1 6562.80A', 'H 1 4861.32A', 'H 1 4340.46A',
                    'H 1 1.87510m', 'H 1 1.28181m', 'H 1 1.09381m']

    # Better to load filter once and save instead
    # repeated calls to SVO
    # fc = set_up_filters(grid)
    fc_flux = FilterCollection(path="./filter_collection_fluxes.hdf5")
    fc_lum = FilterCollection(path="./filter_collection_lums.hdf5")
    fc = [fc_flux, fc_lum]
    combined_filters = FilterCollection(
        path="./filter_collection_combined.hdf5"
        )
    
    # Interpolate grid to filters' wavelength
    grid = Grid(
        args.grid_name,
        grid_dir=args.grid_directory,
        read_lines=True,
        filters=combined_filters
    )

    kern = Kernel()

    if (rank == 0):
        start = time.time()
        create_eagle_shm(args, nthreads=args.nthreads)

        end = time.time()
        print(F"Galaxy numpy memmap generation on rank {rank} took: {end - start:.2f}")

    comm.Barrier()

    lengths = np.load(F'{args.shm_prefix}lengths{args.shm_suffix}.npz')
    s_len, g_len = int(lengths['s_len']), int(lengths['g_len'])

    count = len(fnmatch.filter(
        os.listdir(F"{args.eagle_file}/groups_{args.tag}/"), "eagle_subfind*"
        ))

    comm.Barrier()

    low=0

    n = int((count-low)/args.n_segments)

    print(F"Rank {rank} starting loop execution")

    for ii in range(args.n_segments):
        if ii+1 == args.n_segments:
            high = count
        else:
            high = low + (ii + 1) * n 

        this_low = low + ii*n + rank

        chunks = np.arange(this_low, high, args.total_tasks)
        

        # for ii in range(rank, count, args.total_tasks):
        #     generate_and_save_photometry(
        #         ii, args, grid, fc, spec_keys,
        #         s_len, g_len, kern, tot_chunks=count
        #     )
        generate_and_save_photometry(
                chunks, args, grid, fc, spec_keys,
                line_list, s_len, g_len, kern,
                tot_chunks=count
            )

    print(F"Finished tasks on rank: {rank}")
