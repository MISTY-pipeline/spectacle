from ..modeling.profiles import OpticalDepth1D
from ..modeling.models import Spectral1D
from ..fitting.line_finder import LineFinder1D
from ..utils.detection import region_bounds

import astropy.units as u
import numpy as np
import pytest


@pytest.fixture
def spectral_model():
    spec_mod = Spectral1D([
        OpticalDepth1D(**{'name': 'SiIV1394', 'gamma': 880000000.0, 'f_value': 0.528, 'delta_v': -166.6337 * u.Unit(
            'km/s'), 'column_density': 12.0475, 'v_doppler': 7.3596 * u.Unit('km/s')}),
        OpticalDepth1D(**{'name': 'SiIV1394', 'gamma': 880000000.0, 'f_value': 0.528, 'delta_v': -40.8921 * u.Unit(
            'km/s'), 'column_density': 12.7523, 'v_doppler': 5.6053 * u.Unit('km/s')}),
        OpticalDepth1D(**{'name': 'SiIV1394', 'gamma': 880000000.0, 'f_value': 0.528, 'delta_v': -28.2055 * u.Unit(
            'km/s'), 'column_density': 13.3042, 'v_doppler': 9.433 * u.Unit('km/s')}),
        OpticalDepth1D(**{'name': 'SiIV1394', 'gamma': 880000000.0, 'f_value': 0.528, 'delta_v': -2.7874 * u.Unit(
            'km/s'), 'column_density': 13.2916, 'v_doppler': 4.2219 * u.Unit('km/s')}),
        OpticalDepth1D(**{'name': 'SiIV1394', 'gamma': 880000000.0, 'f_value': 0.528, 'delta_v': 6.9627 * u.Unit(
            'km/s'), 'column_density': 11.9939, 'v_doppler': 5.2558 * u.Unit('km/s')}),
        OpticalDepth1D(**{'name': 'SiIV1394', 'gamma': 880000000.0, 'f_value': 0.528, 'delta_v': 30.309 * u.Unit(
            'km/s'), 'column_density': 12.8783, 'v_doppler': 10.8588 * u.Unit('km/s')})
    ], output='flux')

    return spec_mod


def test_single_line_velocity():
    line1 = OpticalDepth1D(lambda_0=1216 * u.AA, v_doppler=10 * u.km / u.s,
                           column_density=13, delta_v=0 * u.km / u.s)

    # Test finding single line
    spec_mod = Spectral1D([line1], z=0, continuum=0,
                          output='optical_depth')

    # Test line finding in velocity space
    x = np.linspace(-200, 200, 1000) * u.km / u.s
    y = spec_mod(x)  # + 0.001 * (np.random.sample(len(x)) - 0.5)

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
    y = spec_mod(x)  # + 0.001 * (np.random.sample(len(x)) - 0.5)

    line_finder = LineFinder1D(ions=["HI1216"], auto_fit=True,
                               continuum=0, output='optical_depth',
                               threshold=0.05)

    fit_spec_mod = line_finder(x, y)

    assert np.allclose(y, fit_spec_mod(x))


def test_single_line_wavelength():
    line1 = OpticalDepth1D(lambda_0=1216 * u.AA, v_doppler=10 * u.km / u.s,
                           column_density=14, delta_v=0 * u.km / u.s)

    # Test finding single line
    spec_mod = Spectral1D([line1], z=0, continuum=1,
                          output='flux')

    # Test line finding in velocity space
    x = np.linspace(1200, 1250, 2000) * u.AA
    y = spec_mod(x)  # + 0.001 * (np.random.sample(len(x)) - 0.5)

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
    y = spec_mod(x)  # + 0.001 * (np.random.sample(len(x)) - 0.5)

    line_finder = LineFinder1D(ions=["HI1216"], auto_fit=True,
                               continuum=1, output='flux',
                               threshold=0.05)

    fit_spec_mod = line_finder(x, y)

    assert np.allclose(y, fit_spec_mod(x))


def test_detection(spectral_model):
    vel = np.arange(-300, 100, 0.5) * u.Unit('km/s')
    flux = spectral_model(vel)

    reg_bnds = region_bounds(vel, flux, threshold=0.05)

    assert len(reg_bnds) == 6
    assert np.allclose([(508, 565), (583, 620), (256, 277), (534, 565),
                        (583, 605), (642, 679)],
                        list(reg_bnds.keys()))
