import os
import argparse
import time
import fnmatch
from functools import partial

from typing import List, Optional, NamedSpace

import numpy as np
import h5py

from astropy.cosmology import LambdaCDM
from schwimmbad import MultiPool

from synthesizer.grid import Grid
from synthesizer.filters import FilterCollection
from synthesizer.kernel_functions import Kernel
from synthesizer.sed import Sed, combine_list_of_seds
from synthesizer.load_data.load_eagle import load_EAGLE

from utilities import save_dummy_file, get_spectra

# define EAGLE cosmology
cosmo = LambdaCDM(Om0=0.307, Ode0=0.693, H0=67.77, Ob0=0.04825)


def generate_and_save_photometry(
        chunk: int,
        args: NamedSpace,
        grid: Grid,
        fc: FilterCollection,
        spec_keys: List,
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
        kern (Kernel object, optional)
            SPH kernel to use for line-of-sight dust attenuation
        tot_chunks (int)
            total number of files to process
    """

    output_file = F"{args.output}.{chunk}.hdf5"

    gals = load_EAGLE(
            fileloc=args.eagle_file,
            tag=args.tag,
            chunk=chunk,
            numThreads=args.nthreads,
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

    specs = {}
    for key in dat[0].keys():
        specs[key] = combine_list_of_seds([_dat[key] for _dat in dat])

    end = time.time()
    print(f"Spectra generation: {end - start:.2f}")

    # Calculate photometry (observer frame fluxes and luminosities)
    fluxes = {}
    luminosities = {}

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
        "-nthreads",
        type=int,
        required=False,
        help="Number of threads",
        default=2
    )

    parser.add_argument(
        "-chunk",
        type=int,
        required=False,
        help="Eagle file number (chunk) to run",
        default=0
    )

    args = parser.parse_args()
    output_file = F"{args.output}_{args.volume}_{args.tag}"

    grid = Grid(
        args.grid_name,
        grid_dir=args.grid_directory,
        read_lines=False
    )

    kern = Kernel()

    spec_keys = ['stellar', 'intrinsic', 'los']

    # Better to load filter once and save
    # fc = set_up_filters()    
    fc = FilterCollection(path="./filter_collection.hdf5")

    count = len(fnmatch.filter(
        os.listdir(F"{args.eagle_file}/groups_{args.tag}/"), "eagle_subfind*")
        )

    generate_and_save_photometry(
        args.chunk, args, grid, fc, spec_keys, kern, count
        )
