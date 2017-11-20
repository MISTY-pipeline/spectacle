import astropy.units as u
import numpy as np

from spectacle.models import (OpticalDepth, TauProfile)
# from spectacle.spectrum1d import Spectrum1D


def test_compose():
    x = np.arange(5) * u.Unit('km/s')

    hi_line = TauProfile(
        lambda_0=1215.6701 * u.Angstrom, 
        f_value=0.4164, 
        gamma=626500000.0,
        v_doppler=1e6 * u.Unit('cm/s'),
        column_density=1e15 * u.Unit('1/cm2'), 
        delta_v=0 * u.Unit('km/s'),
        delta_lambda=0 * u.Angstrom)

    opt_dep_mod = OpticalDepth()

    

# def test_wavelength_convert():
#     # Define a wavelength array
#     x = np.arange(5) * u.Unit('km/s')

#     # Given the rest frequency of 5 km/s, convert array to velocity space
#     cx = WavelengthConvert(center=2 * u.Unit('km/s'))(x)

#     assert np.isclose(cx.value, [-599584.916, -149896.229, 0, 74948.1145,
#                                  119916.9832])
#     assert cx.unit == u.Unit('Angstrom')
