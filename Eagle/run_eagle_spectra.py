import argparse
import time
from functools import partial
from schwimmbad import MultiPool

from unyt import Myr
from astropy.cosmology import Planck13

from synthesizer.grid import Grid
from synthesizer.filters import FilterCollection
from synthesizer.load_data.load_eagle import load_EAGLE
from synthesizer.kernel_functions import Kernel


def get_spectra(_gal, grid, age_pivot=10. * Myr):
    
    """
    Helper method for spectra generation

    Args:
        _gal (gal type)
        grid (grid type)
        age_pivot (float)
            split between young and old stellar populations, units Myr
    """


    # Skip over galaxies that have no stellar particles
    if _gal.stars.nstars==0:
        print('There are no stars in this galaxy.')
        return None

    spec = {}

    # dtm = _gal.dust_to_metal_vijayan19()

    # Get young pure stellar spectra (integrated)
    young_spec = \
        _gal.stars.get_spectra_incident(grid, young=age_pivot)

    # Get pure stellar spectra for all old star particles
    old_spec_part = \
        _gal.stars.get_particle_spectra_incident(grid, old=age_pivot)

    # Sum and save old and young pure stellar spectra
    old_spec = old_spec_part.sum()

    spec['stellar'] = old_spec + young_spec

    # Get nebular spectra for each star particle
    young_reprocessed_spec_part = \
        _gal.stars.get_particle_spectra_reprocessed(grid, young=age_pivot)

    # Sum and save intrinsic stellar spectra
    young_reprocessed_spec = young_reprocessed_spec_part.sum()


    # Save intrinsic stellar spectra
    spec['intrinsic'] = young_reprocessed_spec + old_spec

    return spec

def set_up_filters():
    
    # define a filter collection object
    # fs = [f"SLOAN/SDSS.{f}" for f in ['u', 'g', 'r', 'i', 'z']]
    # fs += ['GALEX/GALEX.FUV', 'GALEX/GALEX.NUV']
    # fs += [f'Generic/Johnson.{f}' for f in ['U', 'B', 'V', 'J']]
    # fs += [f'2MASS/2MASS.{f}' for f in ['J', 'H', 'Ks']]
    # fs += [f'HST/ACS_HRC.{f}'
    #        for f in ['F435W', 'F606W', 'F775W', 'F814W', 'F850LP']]
    fs = [f'HST/WFC3_IR.{f}'
           for f in ['F098M', 'F105W', 'F110W', 'F125W', 'F140W', 'F160W']]

    fs += [f'JWST/NIRCam.{f}' 
            for f in [
                'F070W', 'F090W', 'F115W', 'F140M', 'F150W',
                'F162M', 'F182M', 'F200W', 'F210M', 'F250M',
                'F277W', 'F300M', 'F356W', 'F360M', 'F410M',
                'F430M', 'F444W', 'F460M', 'F480M']]
    
    # fs += [f'JWST/MIRI.{f}' 
    #         for f in [
    #             'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W',
    #             'F2100W', 'F2550W', 'F560W', 'F770W']]

    # tophats = {
    #     "UV1500": {"lam_eff": 1500, "lam_fwhm": 300},
    #     "UV2800": {"lam_eff": 2800, "lam_fwhm": 300},
    # }

    fc = FilterCollection(
        filter_codes=fs,
        # tophat_dict=tophats,
        new_lam=grid.lam
    )

    return fc

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Run the synthesizer EAGLE pipeline")

    parser.add_argument(
        "tag",
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
        "-nprocs",
        type=int,
        required=False,
        help="Number of threads",
        default=4
    )

    args = parser.parse_args()
    output_file = F"{args.output}_{args.volume}"

    grid = Grid(
        args.grid_name,
        grid_dir=args.grid_directory,
        read_lines=False
    )

    kern = Kernel()

    fc = set_up_filters()
    # fc = FilterCollection(path="filter_collection.hdf5")

    gals = load_EAGLE(
        fileloc=args.eagle_file,
        tag=args.tag,
        chunk=1,
        numThreads=4
    )

    print(f"Number of galaxies: {len(gals)}")

    # If there are no galaxies in this snap, create dummy file
    # if len(gals)==0:
    #     print('No galaxies. Saving dummy file.')
    #     save_dummy_file(args.output, args.region, args.tag,
    #                     [f.filter_code for f in fc])
    #     sys.exit()

    # spec = get_spectra(gals[100], grid)

    # Debugging plot
    # plt.loglog(young_spec.lam, young_spec.lnu, label='young')
    # plt.loglog(old_spec.lam, old_spec.lnu, label='old')
    # plt.loglog(spec['incident'].lam, spec['incident'].lnu, label='combined')
    # plt.loglog(spec['screen'].lam, spec['screen'].lnu, label='screen')
    # plt.loglog(spec['CF00'].lam, spec['CF00'].lnu, label='CF00')
    # plt.loglog(spec['gamma'].lam, spec['gamma'].lnu, label='gamma')
    # plt.loglog(spec['los'].lam, spec['los'].lnu, label='los')
    # plt.xlim(1e1, 9e4)
    # plt.ylim(1e25,1e34)
    # plt.legend()
    # plt.show()
 
    # start = time.time()
    
    _f = partial(get_spectra, grid=grid)
    with MultiPool(args.nprocs) as pool:
        dat = pool.map(_f, gals)

    # # Get rid of Nones (galaxies that don't have stellar particles)
    # mask = np.array(dat)==None
    # dat = np.array(dat)[~mask]

    # # If there are no galaxies in this snap, create dummy file
    # if len(dat)==0:
    #     print('Galaxies have no stellar particles. Saving dummy file.')
    #     save_dummy_file(args.output, args.region, args.tag,
    #                     [f.filter_code for f in fc])
    #     sys.exit()
   
    # # Combine list of dicts into dict with single Sed objects
    # specs = {}
    # for key in dat[0].keys():
    #     specs[key] = combine_list_of_seds([_dat[key] for _dat in dat])

    # end = time.time()
    # print(f'Spectra generation: {end - start:.2f}')
    

    # # Calculate photometry (observer frame fluxes and luminosities)
    # fluxes = {}
    # luminosities = {}
    
    start = time.time()
    for key in dat[0].keys():
        specs[key].get_fnu(cosmo=Planck13, z=gals[0].redshift)
        fluxes[key] = specs[key].get_photo_fluxes(fc)
        luminosities[key] = specs[key].get_photo_luminosities(fc)

    end = time.time()
    print(f'Photometry calculation: {end - start:.2f}')

    # # Save spectra, fluxes and luminosities
    # with h5py.File(args.output, 'w') as hf:

    #     # Use Region/Tag structure
    #     grp = hf.require_group(f'{args.region}/{args.tag}')

    #     # Loop through different spectra / dust models
    #     for key in dat[0].keys():
    #         sbgrp = grp.require_group('SED')
    #         dset = sbgrp.create_dataset(f'{str(key)}', data=specs[key].lnu)
    #         dset.attrs['Units'] = str(specs[key].lnu.units)
    #         # Include wavelength array corresponding to SEDs
    #         if key==np.array(dat[0].keys())[0]:
    #             lam = sbgrp.create_dataset(f'Wavelength', data=specs[key].lam)
    #             lam.attrs['Units'] = str(specs[key].lam.units)

    #         sbgrp = grp.require_group('Fluxes')
    #         # Create separate groups for different instruments
    #         for f in fluxes[key].filters:
    #             dset = sbgrp.create_dataset(
    #                 f'{str(key)}/{f.filter_code}',
    #                 data=fluxes[key][f.filter_code]
    #             )

    #             dset.attrs['Units'] = str(fluxes[key].photometry.units)


    #         sbgrp = grp.require_group('Luminosities')
    #         # Create separate groups for different instruments
    #         for f in luminosities[key].filters:
    #             dset = sbgrp.create_dataset(
    #                 f'{str(key)}/{f.filter_code}',
    #                 data=luminosities[key][f.filter_code]
    #             )

    #             dset.attrs['Units'] = str(luminosities[key].photometry.units)

