import numpy as np
import h5py
import time
from functools import partial

from schwimmbad import MultiPool, SerialPool
from unyt import Myr

from synthesizer.grid import Grid
from synthesizer.sed import Sed
from synthesizer.load_data.load_simba import load_Simba


if __name__ == "__main__":
    
    # grid_name = "bpass-2.2.1-bin_chabrier03-0.1,100.0_cloudy-c17.03"

    # grid_name = 'bc03_chabrier03-0.1,100.hdf5'
    # grid_label = 'BC03'

    grid_name = "fsps-3.2-mist-miles_bpl-0.08,0.5,1,120-1.3,2.3,2.3.hdf5"
    grid_label = 'FSPS'
    
    grid_dir = "../../../data/synthesizer_data/grids/"
    grid = Grid(grid_name, grid_dir=grid_dir, read_lines=False)

    snap = '151'

    # # Temp fix for old BC03 grids
    # grid.spectra['incident'] = grid.spectra['stellar']

    gals = load_Simba(
        directory=("/cosma7/data/dp004/dc-love2/codes/"
                   "simba_dusty_quiescent/data"),
        snap_name=f"snap_m100n1024_{snap}.hdf5",
        caesar_name=f"Groups/m100n1024_{snap}.hdf5",
    )

    def get_spectra(_gal, grid, age_pivot=10. * Myr):
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

        # Get nebular spectra for young stars
        # young_spec = _gal.stars.get_spectra_nebular(grid, young=age_pivot)
        
        young_spec = _gal.stars.get_spectra_incident(grid, young=age_pivot)
        old_spec = _gal.stars.get_spectra_incident(grid, old=age_pivot)

        # Charlot & Fall attenuation model
        young_spec_attenuated = young_spec.apply_attenuation(tau_v=0.33 + 0.67)
        old_spec_attenuated = old_spec.apply_attenuation(tau_v=0.33)
        attenuated_spec = young_spec_attenuated + old_spec_attenuated
        
        return young_spec, old_spec, attenuated_spec
 
    start = time.time()
    
    _f = partial(get_spectra, grid=grid)
    with MultiPool(8) as pool:
        dat = pool.map(_f, gals)

    end = time.time()
    print(f'{end - start:.2f}')

    # Combine into a single Sed object
    young_spec = Sed(lam=dat[0][0].lam, lnu=[_s[0].lnu for _s in dat]) 
    old_spec = Sed(lam=dat[0][0].lam, lnu=[_s[1].lnu for _s in dat]) 
    attenuated_spec = Sed(lam=dat[0][0].lam, lnu=[_s[2].lnu for _s in dat]) 

    with h5py.File('simba.hdf5', 'w') as hf:
        hf.create_dataset(
            f'snap_{snap}/{grid_label}/spectra/young_stellar',
            data=np.array(young_spec.lnu)
        )
        hf.create_dataset(
            f'snap_{snap}/{grid_label}/spectra/old_stellar',
            data=np.array(old_spec.lnu)
        )
        hf.create_dataset(
            f'snap_{snap}/{grid_label}/spectra/attenuated',
            data=np.array(attenuated_spec.lnu)
        )
        hf.create_dataset(
            f'snap_{snap}/{grid_label}/spectra/wavelength',
            data=np.array(young_spec.lam)
        )

