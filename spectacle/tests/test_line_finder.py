from ..modeling.profiles import OpticalDepth1D
from ..modeling.models import Spectral1D
from ..fitting.line_finder import LineFinder1D

import astropy.units as u
import numpy as np


def test_single_line_velocity():
    line1 = OpticalDepth1D(lambda_0=1216 * u.AA, v_doppler=10 * u.km / u.s,
                           column_density=13, delta_v=0 * u.km / u.s)

    # Test finding single line
    spec_mod = Spectral1D([line1], z=0, continuum=0,
                          output='optical_depth')

    # Test line finding in velocity space
    x = np.linspace(-200, 200, 1000) * u.km / u.s
    y = spec_mod(x) #+ 0.001 * (np.random.sample(len(x)) - 0.5)

    line_finder = LineFinder1D(ions=["HI1216"], auto_fit=True,
                               continuum=0, output='optical_depth',
                               threshold=0.05)

    fit_spec_mod = line_finder(x, y)
    fit_spec_mod = line_finder.model_result

    assert np.allclose(y, fit_spec_mod(x))


def test_buried_line_velocity():
    line1 = OpticalDepth1D(lambda_0=1216 * u.AA, v_doppler=10 * u.km / u.s,
                           column_density=13, delta_v=0 * u.km / u.s)
    line2 = OpticalDepth1D("HI1216", delta_v=0 * u.km / u.s,
                           v_doppler=70 * u.km / u.s, )

    # Test finding buried lines
    spec_mod = Spectral1D([line1, line2], z=0, continuum=0,
                          output='optical_depth')

    # Test line finding in velocity space
    x = np.linspace(-200, 200, 1000) * u.km / u.s
    y = spec_mod(x) #+ 0.001 * (np.random.sample(len(x)) - 0.5)

    line_finder = LineFinder1D(ions=["HI1216"], auto_fit=True,
                               continuum=0, output='optical_depth',
                               threshold=0.05)

    fit_spec_mod = line_finder(x, y)
    fit_spec_mod = line_finder.model_result

    assert np.allclose(y, fit_spec_mod(x))


def test_single_line_wavelength():
    line1 = OpticalDepth1D(lambda_0=1216 * u.AA, v_doppler=10 * u.km / u.s,
                           column_density=14, delta_v=0 * u.km / u.s)

    # Test finding single line
    spec_mod = Spectral1D([line1], z=0, continuum=1,
                          output='flux')

    # Test line finding in velocity space
    x = np.linspace(1200, 1250, 2000) * u.AA
    y = spec_mod(x) #+ 0.001 * (np.random.sample(len(x)) - 0.5)

    line_finder = LineFinder1D(ions=["HI1216"], auto_fit=True,
                               continuum=1, output='flux',
                               threshold=0.05)

    fit_spec_mod = line_finder(x, y)

    assert np.allclose(y, fit_spec_mod(x))


def test_buried_line_wavelength():
    line1 = OpticalDepth1D(lambda_0=1216 * u.AA, v_doppler=10 * u.km / u.s,
                           column_density=14, delta_v=0 * u.km / u.s)

    line2 = OpticalDepth1D("HI1216", delta_v=0 * u.km / u.s,
                           v_doppler=100 * u.km / u.s, column_density=14)

    # Test finding single line
    spec_mod = Spectral1D([line1, line2], z=0, continuum=1,
                          output='flux')

    # Test line finding in velocity space
    x = np.linspace(1200, 1250, 2000) * u.AA
    y = spec_mod(x) #+ 0.001 * (np.random.sample(len(x)) - 0.5)

    line_finder = LineFinder1D(ions=["HI1216"], auto_fit=True,
                               continuum=1, output='flux',
                               threshold=0.05)

    fit_spec_mod = line_finder(x, y)

    assert np.allclose(y, fit_spec_mod(x))
