import os
import numpy as np
import h5py
import time
import argparse
from functools import partial
import sys

import matplotlib.pyplot as plt
from unyt import Myr, mas, arcsecond, kpc, Msun, nJy
from astropy.cosmology import Planck13
from scipy.interpolate import interp1d

from schwimmbad import MultiPool

from synthesizer.grid import Grid
from synthesizer.filters import FilterCollection
from synthesizer.imaging import ImageCollection
from synthesizer.load_data.load_flares import load_FLARES
from synthesizer.kernel_functions import Kernel
from synthesizer.conversions import fnu_to_apparent_mag


def get_pixel_hlr(img, single_pix_area, radii_frac=0.5):
    # Get half the total luminosity
    half_l = np.sum(img) * radii_frac

    # Sort pixels into 1D array ordered by decreasing intensity
    sort_1d_img = np.sort(img.flatten())[::-1]
    sum_1d_img = np.cumsum(sort_1d_img)
    cumal_area = np.full_like(sum_1d_img, single_pix_area) * np.arange(
        1, sum_1d_img.size + 1, 1
    )

    npix = np.argmin(np.abs(sum_1d_img - half_l))
    cumal_area_cutout = cumal_area[
        np.max((npix - 10, 0)) : np.min((npix + 10, cumal_area.size - 1))
    ]
    sum_1d_img_cutout = sum_1d_img[
        np.max((npix - 10, 0)) : np.min((npix + 10, cumal_area.size - 1))
    ]

    # Interpolate the arrays for better resolution
    interp_func = interp1d(cumal_area_cutout, sum_1d_img_cutout, kind="linear")
    interp_areas = np.linspace(
        cumal_area_cutout.min(), cumal_area_cutout.max(), 500
    )
    interp_1d_img = interp_func(interp_areas)

    # Calculate radius from pixel area defined using the interpolated arrays
    pix_area = interp_areas[np.argmin(np.abs(interp_1d_img - half_l))]
    hlr = np.sqrt(pix_area / np.pi)

    return hlr


def get_spectra(_gal, grid, age_pivot=10.0 * Myr):
    """
    Helper method for spectra generation

    Args:
        _gal (gal type)
        grid (grid type)
        age_pivot (float)
            split between young and old stellar populations, units Myr
    """

    # Skip over galaxies that have no stellar particles
    if _gal.stars.nstars == 0:
        print("There are no stars in this galaxy.")
        return None

    spec = {}

    _gal.dust_to_metal_vijayan19()

    # Get young pure stellar spectra (integrated)
    young_spec = _gal.stars.get_particle_spectra_incident(
        grid, young=age_pivot
    )

    # Get pure stellar spectra for all old star particles
    old_spec = _gal.stars.get_particle_spectra_incident(grid, old=age_pivot)

    spec["stellar"] = old_spec + young_spec

    # Get nebular spectra for each star particle
    young_reprocessed_spec = _gal.stars.get_particle_spectra_reprocessed(
        grid, young=age_pivot
    )

    # Save intrinsic stellar spectra
    spec["intrinsic"] = young_reprocessed_spec + old_spec

    # Simple screen model
    spec["screen"] = spec["intrinsic"].apply_attenuation(tau_v=0.33)

    # Charlot & Fall attenuation model
    young_spec_attenuated = young_reprocessed_spec.apply_attenuation(
        tau_v=0.33 + 0.67
    )
    old_spec_attenuated = old_spec.apply_attenuation(tau_v=0.33)
    spec["CF00"] = young_spec_attenuated + old_spec_attenuated

    # Gamma model (modified version of Lovell+19)
    gamma = _gal.screen_dust_gamma_parameter()

    young_spec_attenuated = young_reprocessed_spec.apply_attenuation(
        tau_v=gamma * (0.33 + 0.67)
    )
    old_spec_attenuated = old_spec.apply_attenuation(tau_v=gamma * 0.33)

    spec["gamma"] = young_spec_attenuated + old_spec_attenuated

    # LOS model (Vijayan+21)
    tau_v = _gal.calculate_los_tau_v(
        kappa=0.0795, kernel=kern.get_kernel(), force_loop=False
    )

    # plt.hist(np.log10(tau_v))
    # plt.show()

    young_spec_attenuated = young_reprocessed_spec.apply_attenuation(
        tau_v=tau_v + (_gal.stars.metallicities / 0.01)
    )
    old_spec_attenuated = old_spec.apply_attenuation(tau_v=tau_v)

    spec["los"] = young_spec_attenuated + old_spec_attenuated

    return spec


def get_phot(_gal, grid, fc, spectra_type, age_pivot=10.0 * Myr):
    """
    Helper method for spectra generation

    Args:
        _gal (gal type)
        grid (grid type)
        age_pivot (float)
            split between young and old stellar populations, units Myr
    """
    # Skip over galaxies that have no stellar particles
    if _gal.stars.nstars == 0:
        print("There are no stars in this galaxy.")
        return None

    spec = get_spectra(_gal, grid)

    if spec is None:
        return None

    # Get the flux
    spec[spectra_type].get_fnu(cosmo=Planck13, z=_gal.redshift)

    phot = spec[spectra_type].get_photo_fluxes(fc)

    return phot


def get_img_smoothed(
    gal,
    grid,
    fc,
    resolution,
    spectra_type,
    kernel,
    kernel_threshold=1,
    age_pivot=10.0 * Myr,
    width=60 * kpc,
):
    # First get the photometry for this group
    phot = get_phot(gal, grid, fc, spectra_type, age_pivot)

    if phot is None:
        return None

    # Set up image
    gal_img = ImageCollection(resolution=resolution, fov=width)

    # Compute the image
    gal_img.get_imgs_smoothed(
        photometry=phot,
        coordinates=gal.stars.coordinates,
        smoothing_lengths=gal.stars.smoothing_lengths,
        kernel=kernel,
        kernel_threshold=kernel_threshold,
    )

    return gal_img


def set_up_filters(lam):
    fs = [
        f"JWST/NIRCam.{f}"
        for f in [
            "F150W",
            "F200W",
            "F277W",
            "F356W",
            "F444W",
        ]
    ]

    fs += [
        f"JWST/MIRI.{f}"
        for f in [
            "F1000W",
            "F1500W",
            "F2100W",
            "F770W",
        ]
    ]

    tophats = {
        "UV1500": {"lam_eff": 1500, "lam_fwhm": 300},
        "UV2800": {"lam_eff": 2800, "lam_fwhm": 300},
    }

    fc = FilterCollection(filter_codes=fs, tophat_dict=tophats, new_lam=lam)

    return fc


def save_dummy_file(
    h5_file,
    region,
    tag,
    filters,
    keys=["stellar", "intrinsic", "screen", "CF00", "gamma", "los"],
):
    """
    Save a dummy hdf5 file with the expected hdf5 structure but
    containing no data. Useful if a snapshot contains no galaxies.

    Args
        h5_file (str):
            file to be written
        region (str):
            region e.g. '00'
        tag (str):
            snapshot label e.g. '000_z015p000'
        filters:
            filter collection
        keys:
            spectra / dust models
    """

    with h5py.File(h5_file, "w") as hf:
        # Use Region/Tag structure
        grp = hf.require_group(f"{region}/{tag}")

        # Loop through different spectra / dust models
        for key in keys:
            sbgrp = grp.require_group("Images")
            # Create separate groups for different instruments
            for f in filters:
                dset = sbgrp.create_dataset(f"{str(key)}/{f}", data=[])
                dset.attrs["Units"] = "nJy"

            sbgrp = grp.require_group("Sizes")
            # Create separate groups for different instruments
            for f in filters:
                dset = sbgrp.create_dataset(f"{str(key)}/{f}", data=[])
                dset.attrs["Units"] = "pkpc"

            sbgrp = grp.require_group("Fluxes")
            # Create separate groups for different instruments
            for f in filters:
                dset = sbgrp.create_dataset(f"{str(key)}/{f}", data=[])
                dset.attrs["Units"] = "nJy"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the synthesizer FLARES pipeline"
    )

    parser.add_argument(
        "region",
        type=str,
        help="FLARES region",
    )

    parser.add_argument(
        "tag",
        type=str,
        help="FLARES snapshot tag",
    )

    parser.add_argument(
        "-master-file",
        type=str,
        required=False,
        help="FLARES master file",
        default="/cosma7/data/dp004/dc-love2/codes/flares/data/flares.hdf5",
    )

    parser.add_argument(
        "-grid-name",
        type=str,
        required=False,
        help="Synthesizer grid file",
        default="bpass-2.2.1-bin_chabrier03-0.1,100.0_cloudy-c17.03",
    )

    parser.add_argument(
        "-grid-directory",
        type=str,
        required=False,
        help="Synthesizer grid directory",
        default="../../synthesizer_data/grids/",
    )

    parser.add_argument(
        "-output",
        type=str,
        required=False,
        help="Output file",
        default="./flares_passive_obs.hdf5",
    )

    parser.add_argument(
        "-nprocs",
        type=int,
        required=False,
        help="Number of threads",
        default=16,
    )

    parser.add_argument(
        "-spectra-type",
        type=str,
        default="los",
        help="Generate sizes for what spectra?",
    )

    args = parser.parse_args()

    grid = Grid(args.grid_name, grid_dir=args.grid_directory, read_lines=False)

    kern = Kernel()

    # Load the filter collection (either from SVO or disk)
    if not os.path.exists("filter_collection.hdf5"):
        fc = set_up_filters(grid.lam)
        fc.write_filters("filter_collection.hdf5")

    else:
        fc = FilterCollection(path="filter_collection.hdf5")

    gals = load_FLARES(
        master_file=args.master_file,
        region=args.region,
        tag=args.tag,
    )

    # Parse the redshift
    redshift = float(args.tag.split("z")[-1].replace("p", "."))

    # Set up the image resolution
    ang_resolution = (30 * mas).to(arcsecond)
    ang_resolution_radians = ang_resolution.to("radian")
    _resolution = (
        ang_resolution_radians
        * Planck13.angular_diameter_distance(redshift).to("kpc").value
        * kpc
    )
    resolution = (_resolution, _resolution)
    pix_area = _resolution.value**2

    print(f"Number of galaxies: {len(gals)}")

    # Open the file containing the passive galaxy masks
    passive_mask_path = (
        "/cosma7/data/dp004/dc-love2/codes/flares_passive/"
        "analysis/data/select_quiescent_f444w.h5"
    )
    with h5py.File(passive_mask_path, "r") as hdf:
        # Extract the current region and tag group
        grp = hdf[f"{args.tag}/{args.region}"]

        # Get the masses
        smasses = grp["Mstar"][...] * 10**10 * Msun

        # Get the fluxes
        f444w_flux = grp["F444W"][...] * nJy

        # Get the boolean mask for passivity
        passive = grp["quiescent/evolving/100Myr"][...]

    # Convert fluxes to magnitudes
    mags = fnu_to_apparent_mag(f444w_flux)

    # Create mask
    mask = np.logical_and(mags < 26, smasses > 10**9.5)
    mask = np.logical_and(mask, passive)

    # Apply mask
    gals = np.array(gals)[mask]

    # If there are no galaxies in this snap, create dummy file
    if len(gals) == 0:
        print("No galaxies. Saving dummy file.")
        save_dummy_file(
            args.output,
            args.region,
            args.tag,
            [f.filter_code for f in fc],
            keys=(args.spectra_type),
        )
        sys.exit()

    print(f"After cut number of galaxies: {len(gals)}")

    start = time.time()

    _f = partial(
        get_img_smoothed,
        grid=grid,
        fc=fc,
        resolution=resolution,
        spectra_type=args.spectra_type,
        kernel=kern.get_kernel(),
    )
    with MultiPool(args.nprocs) as pool:
        dat = pool.map(_f, gals)

    # Get rid of Nones (galaxies that don't have stellar particles)
    mask = np.array(dat) == None
    dat = np.array(dat)[~mask]

    # If there are no galaxies in this snap, create dummy file
    if len(dat) == 0:
        print("Galaxies have no stellar particles. Saving dummy file.")
        save_dummy_file(
            args.output, args.region, args.tag, [f.filter_code for f in fc]
        )
        sys.exit()

    # Extract the data we'll write out
    imgs = {
        f: np.zeros(
            (
                len(dat),
                dat[0][f].arr.shape[0],
                dat[0][f].arr.shape[1],
            )
        )
        for f in fc.filter_codes
    }
    fluxes = {f: np.zeros(len(dat)) for f in fc.filter_codes}
    sizes = {f: np.zeros(len(dat)) for f in fc.filter_codes}
    for f in fc.filter_codes:
        for i in range(len(dat)):
            imgs[f][i] = dat[i][f].arr
            fluxes[f][i] = np.sum(dat[i][f].arr)
            sizes[f][i] = get_pixel_hlr(imgs[f][i], single_pix_area=pix_area)

    print(f"Image calculation: {time.time() - start:.2f}")

    # Save spectra, fluxes and luminosities
    mode = "w" if not os.path.exists(args.output) else "r+"
    with h5py.File(args.output, mode) as hf:
        # Use Region/Tag structure
        grp = hf.require_group(f"{args.region}/{args.tag}")

        # Loop through different spectra / dust models
        for f in imgs.keys():
            sbgrp = grp.require_group("Images")
            dset = sbgrp.create_dataset(
                f,
                data=imgs[f],
            )
            dset.attrs["Units"] = str(dat[i][f].units)

            sbgrp = grp.require_group("Sizes")
            dset = sbgrp.create_dataset(
                f,
                data=sizes[f],
            )
            dset.attrs["Units"] = "kpc"

            sbgrp = grp.require_group("Fluxes")
            dset = sbgrp.create_dataset(
                f,
                data=fluxes[f],
            )
            dset.attrs["Units"] = str(dat[i][f].units)
