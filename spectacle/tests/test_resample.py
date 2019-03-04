from spectacle import OpticalDepth1D, Spectral1D
from ..analysis import Resample
import numpy as np
import astropy.units as u


def test_resample_simple():
    line1 = OpticalDepth1D(lambda_0=1216 * u.AA)

    # Create model from lambda
    spec_mod_lambda = Spectral1D(line1, output='optical_depth')

    x = np.linspace(-100, 100, 200) * u.km / u.s
    y = spec_mod_lambda(x)

    new_x = np.linspace(-100, 100, 2000) * u.km / u.s

    resample = Resample(new_x)
    new_y = resample(x, y)

    assert np.isclose(np.trapz(y, x.value), np.trapz(new_y, new_x.value))
