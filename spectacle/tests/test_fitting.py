from astropy.modeling.fitting import LevMarLSQFitter
from spectacle.modeling import Spectral1D, OpticalDepth1D
import astropy.units as u
import numpy as np


def test_levmar():
    line1 = OpticalDepth1D(lambda_0=1216 * u.AA, v_doppler=500 * u.km /u.s, column_density=14)
    spec_mod = Spectral1D(line1, continuum=1)
    x = np.linspace(1200, 1225, 1000) * u.Unit('Angstrom')
    y = spec_mod(x)

    fitter = LevMarLSQFitter()
    fit_spec_mod = fitter(spec_mod, x, y)

    assert np.allclose(y, fit_spec_mod(x))