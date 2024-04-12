import numpy as np
import h5py
import time
from functools import partial

from schwimmbad import MultiPool

from synthesizer.grid import Grid
from synthesizer.filters import FilterCollection
from synthesizer.load_data.load_simba import load_Simba
from synthesizer.utils import Lnu_to_M
from synthesizer.dust.attenuation import PowerLaw


if __name__ == "__main__":
    grid_name = "bpass-2.2.1-bin_chabrier03-0.1,100.0_cloudy-c17.03"
    grid_dir = "../../synthesizer_data/grids/"
    grid = Grid(grid_name, grid_dir=grid_dir, read_lines=False)

    # define a filter collection object
    fs = [f"SLOAN/SDSS.{f}" for f in ["u", "g", "r", "i", "z"]]
    # fs += ['GALEX/GALEX.FUV', 'GALEX/GALEX.NUV']
    # fs += [f'Generic/Johnson.{f}' for f in ['U', 'B', 'V', 'J']]
    # fs += [f'2MASS/2MASS.{f}' for f in ['J', 'H', 'Ks']]
    # fs += [f'HST/ACS_HRC.{f}'
    #        for f in ['F435W', 'F606W', 'F775W', 'F814W', 'F850LP']]
    # fs += [f'HST/WFC3_IR.{f}'
    #        for f in ['F098M', 'F105W', 'F110W', 'F125W', 'F140W', 'F160W']]

    # tophats = {
    #     "UV1500": {"lam_eff": 1500, "lam_fwhm": 300},
    #     "UV2800": {"lam_eff": 2800, "lam_fwhm": 300},
    # }

    fc = FilterCollection(
        filter_codes=fs,
        # tophat_dict=tophats,
        new_lam=grid.lam,
    )

    gals = load_Simba(
        directory=("/cosma7/data/dp004/dc-love2/codes/" "simba_dusty_quiescent/data"),
        snap_name="snap_m100n1024_144.hdf5",
        caesar_name="Groups/m100n1024_144.hdf5",
    )

    def get_spectra(_gal, grid, fc, age_pivot=10.0):
        """
        Helper method for spectra generation

        Args:
            _gal (gal type)
            grid (grid type)
            fc (FilterCollection type)
            age_pivot (float)
                split between young and old stellar populations, units Myr
        """

        spec = {}
        # young_spec = _gal.stars.get_spectra_incident(grid, young=10)

        # Get nebular spectra for young stars
        young_spec = _gal.stars.get_spectra_nebular(grid, young=age_pivot)
        old_spec = _gal.stars.get_spectra_incident(grid, old=age_pivot)

        # Save intrinsic stellar spectra
        spec["stellar"] = young_spec + old_spec

        ## Screen model variations
        tau_v_arr = np.linspace(0.05, 0.5, 20)
        power_law_slope = np.linspace(-0.5, -2.0, 20)

        mags = [None] * (len(tau_v_arr) * len(power_law_slope))

        for i, tau_v in enumerate(tau_v_arr):
            for j, pl_slope in enumerate(power_law_slope):
                _spec = spec["stellar"].apply_attenuation(
                    tau_v=tau_v,
                    dust_curve=PowerLaw(slope=pl_slope),
                )
                _spec.get_fnu0()
                fluxes = _spec.get_broadband_luminosities(fc)

                index = i * len(tau_v_arr) + j

                mags[index] = [Lnu_to_M(v) for k, v in fluxes.items()]

        return mags

    start = time.time()

    _f = partial(get_spectra, grid=grid, fc=fc)
    with MultiPool(8) as pool:
        dat = pool.map(_f, gals)

    end = time.time()
    print(f"{end - start:.2f}")

    with h5py.File("simba_screen_mags.hdf5", "w") as hf:
        hf.create_dataset("mags", data=np.array(mags))
