import os
import gc
import argparse
import time
import fnmatch
from functools import partial

from collections import namedtuple
from typing import List, Optional, Dict

import numpy as np
import h5py

from astropy.cosmology import LambdaCDM
from schwimmbad import MultiPool

from synthesizer.grid import Grid
from synthesizer.filters import FilterCollection
from synthesizer.kernel_functions import Kernel
from synthesizer.sed import Sed
from synthesizer.load_data.load_eagle import load_EAGLE
from unyt import Angstrom, erg, s

from utilities import save_dummy_file, get_spectra, get_lines

# define EAGLE cosmology
cosmo = LambdaCDM(Om0=0.307, Ode0=0.693, H0=67.77, Ob0=0.04825)


def generate_and_save_photometry(
        args: namedtuple,
        grid: Grid,
        fc: FilterCollection,
        spec_keys: List,
        line_list: List,
        kern: Optional[Kernel] = None,
        tot_chunks: int = 1535
        ) -> None:
    """
    Read in the eagle galaxy data for the chunk subfind file and
    write out the required photometry information in spec_keys
    Arguments:
        args (NamedSpace):
            parser arguments passed on to this job
        grid (Grid object)
            SPS grid object to generate the spectra
        fc (FilterCollection object)
            filters to generate the photometry
        spec_keys (list)
            list of galaxy spectra type to generate
        spec_keys (list)
            list of galaxy spectra type to generate
        kern (Kernel object, optional)
            SPH kernel to use for line-of-sight dust attenuation
        tot_chunks (int)
            total number of files to process
    """

    fc_flux, fc_lum = fc

    output_file = F"{args.output}.{args.chunk}.hdf5"

    gals = load_EAGLE(
            fileloc=args.eagle_file,
            tag=args.tag,
            chunk=args.chunk,
            numThreads=args.nthreads,
            tot_chunks=tot_chunks,
    )

    print(f"Number of galaxies: {len(gals)}")

    # If there are no galaxies in this snap, create dummy file
    if gals == []:
        print('No galaxies. Saving dummy file.')
        save_dummy_file(output_file,
                        filters=fc,
                        line_list=line_list,
                        keys=spec_keys,
                        )
        return None

    start = time.time()
    _f = partial(get_spectra, grid=grid, spec_keys=spec_keys, kern=kern)
    with MultiPool(args.nthreads) as pool:
        dat = np.array(pool.map(_f, gals))

    gals = np.array(dat[:,1], dtype=object)
    dat = list(dat[:,0])
    gc.collect()
    # # galaxies that don't have stellar particles
    mask = np.array(dat) == {}
    if len(gals) == np.sum(mask):
        print('No stars in galaxies. Saving dummy file.')
        save_dummy_file(output_file,
                        filters=fc,
                        line_list=line_list,
                        keys=spec_keys,
                        gal_num = len(gals)
                        )
        return None

    if np.sum(mask)>0:
        ok = np.where(~mask)[0]
        gals = gals[ok]
    
        # Assign null SEDs to galaxies with no stars
        nostars = np.where(mask)[0]
        null_sed = Sed(lam=grid.lam)
        null_dict = {ii: null_sed for ii in spec_keys}
        dat = np.array(dat, dtype=object)
        dat[nostars] = null_dict

    specs: Dict = {}
    print(args.chunk, dat)
    for key in dat[0].keys():
        specs[key] = Sed(grid.lam, np.array([_dat[key].lnu for _dat in dat]))

    end = time.time()
    print(f"Spectra generation: {end - start:.2f}")

    # Calculate photometry (observer frame fluxes and luminosities)
    fluxes: Dict = {}
    luminosities: Dict = {}

    start = time.time()
    for key in dat[0].keys():
        specs[key].get_fnu(cosmo=cosmo,
                           z=float(args.tag[5:].replace("p", ".")))
        fluxes[key] = specs[key].get_photo_fluxes(fc_flux, verbose=False)
        luminosities[key] = specs[key].get_photo_luminosities(fc_lum, verbose=False)

    end = time.time()
    print(f"Photometry calculation: {end - start:.2f}")

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
        dat = pool.map(_f, gals)

    # make temporary array to store the line outputs
    # of zero star galaxies
    tmp_lum = np.zeros((len(mask), len(line_list))) * erg/s
    tmp_ew = np.zeros((len(mask), len(line_list))) * Angstrom
    ok = np.where(~mask)[0]
    for ii, key in enumerate(line_keys):
        for jj, _line in enumerate(line_list):
            tmp_lum[:,jj][ok] = np.array([kk[key][_line]._luminosity for kk in dat])
            tmp_ew[:,jj][ok] = np.array([kk[key][_line]._equivalent_width for kk in dat])

        lines_lum[key] = tmp_lum
        lines_ew[key] = tmp_ew

    end = time.time()
    print(
        f"Line calculation for chunk {args.chunk} took: {end - start:.2f}"
        )

    # # Save spectra, fluxes and luminosities
    with h5py.File(F'{args.output}.{args.chunk}.hdf5', 'w') as hf:

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
                    data=fluxes[key][f.filter_code],
                    dtype=np.float32
                )

                dset.attrs['Units'] = str(fluxes[key].photometry.units)

            grp = hf.require_group('Luminosities')
            # Create separate groups for different instruments
            for f in luminosities[key].filters:
                dset = grp.create_dataset(
                    f'{str(key)}/{f.filter_code}',
                    data=luminosities[key][f.filter_code],
                    dtype=np.float32
                )

                dset.attrs['Units'] = str(luminosities[key].photometry.units)
        
            if key != 'stellar':
                grp = hf.require_group('Lines')
                # Create separate groups for different nebular lines
                for jj, f in enumerate(line_list):
                    dset = grp.create_dataset(
                    f'{str(key)}/{f}/Luminosities',
                    data=lines_lum[key][:,jj],
                    dtype=np.float32
                    )

                    dset.attrs['Units'] = str(lines_lum[key].units)

                    dset = grp.create_dataset(
                    f'{str(key)}/{f}/EWs',
                    data=lines_lum[key][:,jj],
                    dtype=np.float32
                    )

                    dset.attrs['Units'] = str(lines_lum[key].units)


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
        "-eagle-file",
        type=str,
        required=False,
        help="EAGLE file",
        default='/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data'
    )

    parser.add_argument(
        "-volume",
        type=str,
        required=False,
        help="EAGLE volume name",
        default='REF_100'
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
        "-chunk",
        type=int,
        required=False,
        help="Which hdf5 chunk to run",
        default=1
    )

    parser.add_argument(
        "-nthreads",
        type=int,
        required=False,
        help="Number of threads",
        default=2
    )

    parser.add_argument(
        "-node-name",
        type=str,
        required=False,
        help="Name of node",
        default=''
    )

    args = parser.parse_args()

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

    count = len(fnmatch.filter(
                os.listdir(F"{args.eagle_file}/groups_{args.tag}/"), "eagle_subfind*"
    ))

    generate_and_save_photometry(
        args=args,
        grid=grid,
        fc=fc,
        spec_keys=spec_keys,
        line_list=line_list,
        kern=kern,
        tot_chunks=count
    )

   

