import os
import time
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

from synthesizer.load_data.load_eagle import read_array, get_age
from synthesizer.grid import Grid
from synthesizer.filters import FilterCollection
from synthesizer.kernel_functions import Kernel
from synthesizer.sed import Sed, combine_list_of_seds
from synthesizer.load_data.load_eagle import load_EAGLE_shm

from utilities import (
    save_dummy_file, get_spectra
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


def generate_and_save_photometry(
        chunk: int,
        args: namedtuple,
        grid: Grid,
        fc: FilterCollection,
        spec_keys: List,
        s_len: int,
        g_len: int,
        kern: Optional[Kernel] = None,
        tot_chunks: int = 1535
        ) -> None:
    """
    Read in the eagle galaxy data for the chunk subfind file and
    write out the required photometry information in spec_keys
    Arguments:
        chunk (int)
            file number to process
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

    output_file = F"{args.output}.{chunk}.hdf5"

    gals = load_EAGLE_shm(
            fileloc=args.eagle_file,
            tag=args.tag,
            s_len=s_len,
            g_len=g_len,
            args=args,
            numThreads=args.nthreads,
            chunk=chunk,
            tot_chunks=tot_chunks,
    )

    print(f"Number of galaxies: {len(gals)}")

    # If there are no galaxies in this snap, create dummy file
    if len(gals) == 0:
        print('No galaxies. Saving dummy file.')
        save_dummy_file(output_file,
                        [f.filter_code for f in fc],
                        keys=spec_keys)
        return None

    start = time.time()
    _f = partial(get_spectra, grid=grid, spec_keys=spec_keys, kern=kern)
    with MultiPool(args.nthreads) as pool:
        dat = pool.map(_f, gals)

    # # galaxies that don't have stellar particles
    mask = np.array(dat) == {}
    if len(gals) == np.sum(mask):
        print('No stars in galaxies. Saving dummy file.')
        save_dummy_file(output_file,
                        [f.filter_code for f in fc],
                        keys=spec_keys)
        return None

    null_sed = Sed(lam=dat[np.where(~mask)[0][0]]['stellar'].wavelength)
    for jj in np.where(mask)[0]:
        for kk in spec_keys:
            dat[jj][kk] = null_sed

    specs: Dict = {}
    for key in dat[0].keys():
        specs[key] = combine_list_of_seds([_dat[key] for _dat in dat])

    end = time.time()
    print(f"Spectra generation: {end - start:.2f}")

    # Calculate photometry (observer frame fluxes and luminosities)
    fluxes: Dict = {}
    luminosities: Dict = {}

    start = time.time()
    for key in dat[0].keys():
        specs[key].get_fnu(cosmo=cosmo, z=gals[0].redshift)
        fluxes[key] = specs[key].get_photo_fluxes(fc)
        luminosities[key] = specs[key].get_photo_luminosities(fc)

    end = time.time()
    print(f"Photometry calculation: {end - start:.2f}")

    # # Save spectra, fluxes and luminosities
    with h5py.File(output_file, 'w') as hf:

        # Loop through different spectra / dust models
        for key in dat[0].keys():
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
                    data=fluxes[key][f.filter_code]
                )

                dset.attrs['Units'] = str(fluxes[key].photometry.units)

            grp = hf.require_group('Luminosities')
            # Create separate groups for different instruments
            for f in luminosities[key].filters:
                dset = grp.create_dataset(
                    f'{str(key)}/{f.filter_code}',
                    data=luminosities[key][f.filter_code]
                )

                dset.attrs['Units'] = str(luminosities[key].photometry.units)


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

    args = parser.parse_args()
    output_file = F"{args.output}_{args.volume}_{args.tag}"

    spec_keys = ['stellar', 'intrinsic', 'los']

    grid = Grid(
        args.grid_name,
        grid_dir=args.grid_directory,
        read_lines=False
    )

    kern = Kernel()

    if (rank == 0):
        create_eagle_shm(args)

    comm.Barrier()

    lengths = np.load(F'{args.shm_prefix}lengths{args.shm_suffix}.npz')
    s_len, g_len = int(lengths['s_len']), int(lengths['g_len'])

    # Better to load filter once and save instead
    # repeated calls to SVO
    # fc = set_up_filters()
    fc = FilterCollection(path="./filter_collection.hdf5")

    count = len(fnmatch.filter(
        os.listdir(F"{args.eagle_file}/groups_{args.tag}/"), "eagle_subfind*"
        ))

    comm.Barrier()

    print(F"Rank {rank} starting loop execution")

    for ii in range(rank, count, args.total_tasks):
        generate_and_save_photometry(
            ii, args, grid, fc, spec_keys,
            s_len, g_len, kern, tot_chunks=count
        )

    print(F"Finished tasks on rank: {rank}")
