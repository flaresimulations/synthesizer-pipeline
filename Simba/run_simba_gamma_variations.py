import numpy as np
import h5py
import time
from functools import partial

from schwimmbad import MultiPool

from synthesizer.grid import Grid
from synthesizer.filters import FilterCollection
from synthesizer.load_data.load_simba import load_Simba
from synthesizer.utils import Lnu_to_M


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

        # Gamma model variations
        gamma_min_arr = [0.01, 0.02, 0.04, 0.04, 0.05, 0.06]
        gamma_max_arr = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        tau_V_ISM_arr = [0.1, 0.2, 0.3, 0.4]
        tau_V_BC_arr = [0.5, 0.6, 0.7, 0.8]

        mags = [None] * (
            len(gamma_min_arr)
            * len(gamma_max_arr)
            * len(tau_V_ISM_arr)
            * len(tau_V_BC_arr)
        )

        for i, gamma_min in enumerate(gamma_min_arr):
            for j, gamma_max in enumerate(gamma_max_arr):
                for k, tau_V_ISM in enumerate(tau_V_ISM_arr):
                    for l, tau_V_BC in enumerate(tau_V_BC_arr):
                        gamma = _gal.screen_dust_gamma_parameter(
                            gamma_min=gamma_min,
                            gamma_max=gamma_max,
                        )

                        young_spec_attenuated = young_spec.apply_attenuation(
                            tau_v=float(gamma * (tau_V_ISM + tau_V_BC))
                        )
                        old_spec_attenuated = old_spec.apply_attenuation(
                            tau_v=float(gamma * tau_V_ISM)
                        )

                        spec = young_spec_attenuated + old_spec_attenuated
                        spec.get_fnu0()
                        fluxes = spec.get_broadband_luminosities(fc)

                        index = (
                            i
                            * len(gamma_max_arr)
                            * len(tau_V_BC_arr)
                            * len(tau_V_ISM_arr)
                            + j * len(tau_V_BC_arr) * len(tau_V_ISM_arr)
                            + k * len(tau_V_BC_arr)
                            + l
                        )

                        mags[index] = [Lnu_to_M(v) for k, v in fluxes.items()]

        return mags

    start = time.time()

    _f = partial(get_spectra, grid=grid, fc=fc)
    with MultiPool(8) as pool:
        dat = pool.map(_f, gals)

    end = time.time()
    print(f"{end - start:.2f}")

    with h5py.File("simba.hdf5", "w") as hf:
        hf.create_dataset("mags", data=np.array(mags))
