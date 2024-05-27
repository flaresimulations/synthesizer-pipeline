import numpy as np
import h5py
from unyt import Myr

from typing import List, Optional, Dict, Tuple
from unyt import unyt_array

from synthesizer.grid import Grid
from synthesizer.filters import FilterCollection
from synthesizer.particle.galaxy import Galaxy
from synthesizer.sed import Sed
from synthesizer.line import Line
from synthesizer.kernel_functions import Kernel


def set_up_filters(grid: Grid) -> FilterCollection:
    # define a filter collection object

    fs = []

    fs += [f"SLOAN/SDSS.{f}" for f in ['u', 'g', 'r', 'i', 'z']]

    fs += ['GALEX/GALEX.FUV', 'GALEX/GALEX.NUV']

    # fs += [f'Generic/Johnson.{f}' for f in ['U', 'B', 'V', 'J']]

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

    fc_flux = FilterCollection(
        filter_codes=fs,
        new_lam=grid.lam
    )
    fc_flux.write_filters(path="./filter_collection_fluxes.hdf5")

    fc_lum = FilterCollection(
        filter_codes=[f'Generic/Johnson.{f}' for f in ['U', 'B', 'V', 'J']],
        tophat_dict=tophats,
        new_lam=grid.lam
    )
    fc_lum.write_filters(path="./filter_collection_lums.hdf5")

    combined_filters = FilterCollection(
        filter_codes=[f'Generic/Johnson.{f}' for f in ['U', 'B', 'V', 'J']]+fs,
        tophat_dict=tophats,
        new_lam=grid.lam
    )
    combined_filters.write_filters(path="./filter_collection_combined.hdf5")

    fc = [fc_flux, fc_lum]

    return fc


def save_dummy_file(
    h5_file: str,
    filters: List[FilterCollection],
    line_list: List = ['H 1 6562.80A', 'H 1 4861.32A', 'H 1 4340.46A'],
    keys: List = ["stellar", "intrinsic", "los"],
    gal_num: int = 0,
) -> None:
    """
    Save a dummy hdf5 file with the expected hdf5 structure but
    containing no data. Useful if a snapshot contains no galaxies.

    Args
        h5_file (str):
            file to be written
        filters:
            filter collection
        line_list (list):
            list of lines to save
        keys (list):
            spectra / dust models
        gal_num (int):
            Number of galaxies in this snap
    """

    fc_flux, fc_lum = filters
    
    if gal_num == 0:
        out: List = []
    else:
        out = np.zeros(gal_num, dtype=np.float32)

    with h5py.File(h5_file, "w") as hf:

        # Loop through different spectra / dust models
        for key in keys:
            # grp = hf.require_group("SED")
            # dset = grp.create_dataset(f"{str(key)}", data=[])
            # dset.attrs["Units"] = "erg/(Hz*s)"

            grp = hf.require_group("Fluxes")
            # Create separate groups for different instruments
            for f in fc_flux:
                dset = grp.create_dataset(f"{str(key)}/{f}", data=out)
                dset.attrs["Units"] = "erg/(cm**2*s)"

            grp = hf.require_group("Luminosities")
            # Create separate groups for different instruments
            for f in fc_lum:
                dset = grp.create_dataset(f"{str(key)}/{f}", data=out)
                dset.attrs["Units"] = "erg/s"


            if key != 'stellar':
                grp = hf.require_group('Lines')
                for f in line_list:
                    dset = grp.create_dataset(f"{str(key)}/{f}/Luminosities", data=out)
                    dset.attrs["Units"] = "erg/s"

                    dset = grp.create_dataset(f"{str(key)}/{f}/EWs", data=out)
                    dset.attrs["Units"] = "Angstrom"



def get_spectra(
        _gal: Galaxy,
        grid: Grid,
        spec_keys: List,
        age_pivot: unyt_array = 10. * Myr,
        kern: Optional[Kernel] = None,
        Znorm: float = 0.01,
        kappa_ISM: float = 0.0795,
        kappa_BC: float = 1.,
        verbose=False
        ) -> Tuple[Dict[str, Sed], Galaxy]:
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
        kern (kernel function)
            the kernel to use for calculating LOS density
        Znorm (float)
            normalisation factor for birth cloud metallicity
        kappa_ISM (float)
            scale factor for diffuse dust
        kappa_BC (float)
            scale factor for birth cloud dust
    """

    spec: Dict = {}

    nstars = _gal.stars.nstars

    # Skip over galaxies that have no stellar particles
    if nstars==0:
        if verbose:
            print('There are no stars in this galaxy.')
        return spec, _gal
    
    # Nebular processed particle spectra
    reproc_spec_part = _gal.stars.get_particle_spectra_reprocessed(grid)

    # Pure stellar spectra
    if 'stellar' in spec_keys:
        spec['stellar'] = (
             _gal.stars.get_spectra_incident(grid)
        )

    # Stellar spectra with nebular emission
    if 'intrinsic' in spec_keys:
        spec['intrinsic'] = (
            reproc_spec_part.sum()
        )

    # Dust attenuated spectra following LOS model from
    # FLARES-II paper
    if 'los' in spec_keys:
        if _gal.gas.nparticles==0:
            if 'intrinsic' in spec:
                spec['los'] = spec['intrinsic']
            else:
                spec['los'] = reproc_spec_part.sum()
        else:
            # LOS model (Vijayan+21)
            
            # Calculate DTM using Vijayan+19 Lgal SAM
            _gal.dust_to_metal_vijayan19()
            
            tau_v_BC = np.zeros(nstars)
            mask = (_gal.stars.ages <= age_pivot)
            tau_v_BC[mask] = kappa_BC * (_gal.stars.metallicities[mask] / Znorm)
            
            tau_v = _gal.calculate_los_tau_v(
                kappa=kappa_ISM, kernel=kern.get_kernel(), force_loop=False
            )

            spec["los"] = reproc_spec_part.apply_attenuation(
                tau_v = tau_v + kappa_BC * (_gal.stars.metallicities / Znorm)
            )

    return spec, _gal


def get_lines(
        _gal: Galaxy,
        grid: Grid,
        line_keys: List,
        line_list: List,
        age_pivot: unyt_array = 10. * Myr,
        kern: Optional[Kernel] = None,
        Znorm: float = 0.01,
        kappa_ISM: float = 0.0795,
        kappa_BC: float = 1.,
        verbose=False
        ) -> Dict[str, Line]:
    """
    Helper method for spectra generation

    Args:
        _gal (Galaxy object)
            Galaxy object to get spectra of
        grid (Grid object)
            SPS grid to use
        line_keys (list)
            spectra / dust models
        line_list (list)
            list of lines
        age_pivot (unyt array)
            split between young and old stellar populations, units Myr
        kern (kernel function)
            the kernel to use for calculating LOS density
        Znorm (float)
            normalisation factor for birth cloud metallicity
        kappa_ISM (float)
            scale factor for diffuse dust
        kappa_BC (float)
            scale factor for birth cloud dust
    """

    line: Dict = {}

    nstars = _gal.stars.nstars

    # Skip over galaxies that have no stellar particles
    if nstars==0:
        if verbose:
            print('There are no stars in this galaxy.')
        return line
    
    # Line emission no dust
    if 'intrinsic' in line_keys:
        line['intrinsic'] = _gal.stars.get_line_intrinsic(
        grid, line_ids=line_list
        )
    
    # Dust attenuated spectra following LOS model from
    # FLARES-II paper
    if 'los' in line_keys:
        if _gal.gas.nparticles==0:
            if 'intrinsic' in line:
                line['los'] = line['intrinsic']
            else:
                line['los'] =  _gal.stars.get_particle_line_intrinsic(
                    grid, line_ids=line_list
                )
        else:
            # LOS model (Vijayan+21)
            
            tau_v_BC = np.zeros(nstars)
            mask = (_gal.stars.ages <= age_pivot)
            tau_v_BC[mask] = kappa_BC * (_gal.stars.metallicities[mask] / Znorm)
            
            if _gal.stars.tau_v is not None:
                # if stars.tau_v is already calculated
                line['los'] = _gal.stars.get_particle_line_attenuated(
                    grid,
                    line_ids=line_list,
                    tau_v_nebular=_gal.stars.tau_v + tau_v_BC,
                    tau_v_stellar=_gal.stars.tau_v,
                )  
            else:
                # Calculate DTM using Vijayan+19 Lgal SAM
                _gal.dust_to_metal_vijayan19()
                                
                tau_v = _gal.calculate_los_tau_v(
                    kappa=kappa_ISM, kernel=kern.get_kernel(), force_loop=False
                )

                line['los'] = _gal.stars.get_particle_line_attenuated(
                    grid,
                    line_ids=line_list,
                    tau_v_nebular=tau_v + tau_v_BC,
                    tau_v_stellar=tau_v,
                )  
    
    return line
