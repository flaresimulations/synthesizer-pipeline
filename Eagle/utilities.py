import h5py
from unyt import Myr

from typing import List, Optional, Dict
from unyt import unyt_array

from synthesizer.grid import Grid
from synthesizer.filters import FilterCollection
from synthesizer.particle.galaxy import Galaxy
from synthesizer.sed import Sed
from synthesizer.kernel_functions import Kernel


def set_up_filters(grid: Grid) -> FilterCollection:
    # define a filter collection object

    fs = []

    fs += [f"SLOAN/SDSS.{f}" for f in ['u', 'g', 'r', 'i', 'z']]

    fs += ['GALEX/GALEX.FUV', 'GALEX/GALEX.NUV']

    fs += [f'Generic/Johnson.{f}' for f in ['U', 'B', 'V', 'J']]

    fs += [f'HST/ACS_HRC.{f}'
           for f in ['F435W', 'F606W', 'F775W', 'F814W', 'F850LP']]

    fs += [f'HST/WFC3_IR.{f}'
           for f in ['F098M', 'F105W', 'F110W', 'F125W', 'F140W', 'F160W']]

    fs += [f'JWST/NIRCam.{f}'
           for f in [
                'F070W', 'F090W', 'F115W', 'F140M', 'F150W',
                'F162M', 'F182M', 'F200W', 'F210M', 'F250M',
                'F277W', 'F300M', 'F356W', 'F360M', 'F410M',
                'F430M', 'F444W', 'F460M', 'F480M']]

    fs += [f'JWST/MIRI.{f}'
           for f in [
                'F560W', 'F770W', 'F1000W', 'F1130W', 'F1280W',
                'F1500W', 'F1800W', 'F2100W', 'F2550W']]

    fs += [f'Euclid/NISP.{f}'
           for f in ['Y', 'J', 'H']]

    fs += [f'LSST/LSST.{f}'
           for f in ['u', 'g', 'r', 'i', 'z', 'y']]

    fs += [f'Paranal/VISTA.{f}'
           for f in ['Z', 'NB980', 'NB990', 'Y', 'NB118', 'J',
                     'H', 'Ks']]

    tophats = {
        "UV1500": {"lam_eff": 1500, "lam_fwhm": 300},
        "UV2800": {"lam_eff": 2800, "lam_fwhm": 300},
    }

    fc = FilterCollection(
        filter_codes=fs,
        tophat_dict=tophats,
        new_lam=grid.lam
    )

    return fc


def save_dummy_file(
    h5_file: str,
    filters: FilterCollection,
    keys: List = ["stellar", "intrinsic", "screen", "CF00", "gamma", "los"],
) -> None:
    """
    Save a dummy hdf5 file with the expected hdf5 structure but
    containing no data. Useful if a snapshot contains no galaxies.

    Args
        h5_file (str):
            file to be written
        filters:
            filter collection
        keys:
            spectra / dust models
    """

    with h5py.File(h5_file, "w") as hf:

        # Loop through different spectra / dust models
        for key in keys:
            # grp = hf.require_group("SED")
            # dset = grp.create_dataset(f"{str(key)}", data=[])
            # dset.attrs["Units"] = "erg/(Hz*s)"

            grp = hf.require_group("Fluxes")
            # Create separate groups for different instruments
            for f in filters:
                dset = grp.create_dataset(f"{str(key)}/{f}", data=[])
                dset.attrs["Units"] = "erg/(cm**2*s)"

            grp = hf.require_group("Luminosities")
            # Create separate groups for different instruments
            for f in filters:
                dset = grp.create_dataset(f"{str(key)}/{f}", data=[])
                dset.attrs["Units"] = "erg/s"


def get_spectra(
        _gal: Galaxy,
        grid: Grid,
        spec_keys: List,
        age_pivot: unyt_array = 10. * Myr,
        kern: Optional[Kernel] = None
        ) -> Dict[str, Sed]:
    """
    Helper method for spectra generation

    Args:
        _gal (Galaxy object)
            Galaxy object to get spectra of
        grid (Grid object)
            SPS grid to use
        spec_keys (list)
            spectra / dust models
        age_pivot (unyt array)
            split between young and old stellar populations, units Myr
    """

    spec: Dict = {}

    # Skip over galaxies that have no stellar particles
    if _gal.stars.nstars==0:
        # print('There are no stars in this galaxy.')
        return spec
    
    # Particle spectra with age>age_pivot
    old_spec_part = _gal.stars.get_particle_spectra_incident(
        grid, old=age_pivot
        )
    # Nebular reprocessed particle spectra with age<age_pivot
    young_reprocessed_spec_part = _gal.stars.get_particle_spectra_reprocessed(
        grid, young=age_pivot
        )

    # Pure stellar spectra
    if 'stellar' in spec_keys:
        spec['stellar'] = (
             _gal.stars.get_spectra_incident(grid, young=age_pivot)
             + old_spec_part.sum()
        )

    # Stellar spectra with nebular emission
    if 'intrinsic' in spec_keys:
        spec['intrinsic'] = (
            young_reprocessed_spec_part.sum()
            + old_spec_part.sum()
        )

    # Dust attenuated spectra following LOS model from
    # FLARES-II paper
    if 'los' in spec_keys:
        if _gal.gas.nparticles==0:
            if 'intrinsic' in spec:
                spec['los'] = spec['intrinsic']
            else:
                spec['los'] = (
                    young_reprocessed_spec_part.sum()
                    + old_spec_part.sum()
                )
        else:
            # LOS model (Vijayan+21)
            tau_v = _gal.calculate_los_tau_v(
                kappa=0.0795, kernel=kern.get_kernel(), force_loop=False
            )

            young_spec_attenuated = (
                young_reprocessed_spec_part.apply_attenuation(
                    tau_v=tau_v + (_gal.stars.metallicities / 0.01)
                )
            )
            old_spec_attenuated = old_spec_part.apply_attenuation(tau_v=tau_v)

            spec["los"] = (
                young_spec_attenuated.sum()
                + old_spec_attenuated.sum()
            )

    return spec
