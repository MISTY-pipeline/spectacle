from spectacle.modeling import Spectral1D, OpticalDepth1D
import astropy.units as u
import numpy as np
from astropy.tests.helper import assert_quantity_allclose


def test_line_stats():
    line1 = OpticalDepth1D(lambda_0=1216 * u.AA, v_doppler=7 * u.km/u.s,
                           column_density=13, delta_v=0 * u.km/u.s)
    line2 = OpticalDepth1D("HI1216", delta_v=30 * u.km/u.s,
                           v_doppler=12 * u.km/u.s, column_density=13)

    spec_mod = Spectral1D([line1, line2], continuum=1, output='flux')

    x = np.linspace(-200, 200, 1000) * u.km / u.s

    line_stats = spec_mod.line_stats(x)

    assert_quantity_allclose(line_stats['ew'][0], 0.05040091274475814 * u.AA)
    assert_quantity_allclose(line_stats['dv90'][0], 15.21521521521521 * u.km/u.s)
    assert_quantity_allclose(line_stats['fwhm'][0], 0.04731861 * u.AA)

    assert_quantity_allclose(line_stats['ew'][1], 0.08632151159859157 * u.AA)
    assert_quantity_allclose(line_stats['dv90'][1], 26.426426426426445 * u.km/u.s)
    assert_quantity_allclose(line_stats['fwhm'][1], 0.08107079 * u.AA)


def test_region_stats():
    line1 = OpticalDepth1D(lambda_0=1216 * u.AA, v_doppler=7 * u.km/u.s,
                           column_density=13, delta_v=0 * u.km/u.s)
    line2 = OpticalDepth1D("HI1216", delta_v=30 * u.km/u.s,
                           v_doppler=12 * u.km/u.s, column_density=13)

    spec_mod = Spectral1D([line1, line2], continuum=1, output='flux')

    x = np.linspace(-200, 200, 1000) * u.km / u.s

    region_stats = spec_mod.region_stats(x, rest_wavelength=1216 * u.AA,
                                         abs_tol=0.05)

    assert_quantity_allclose(region_stats['region_start'], -12.612612612612594 * u.km/u.s)
    assert_quantity_allclose(region_stats['region_end'], 48.648648648648674 * u.km/u.s)
    assert_quantity_allclose(region_stats['ew'], 0.12297380252108686 * u.AA)
    assert_quantity_allclose(region_stats['dv90'], 46.44644644644646 * u.km/u.s)
    assert_quantity_allclose(region_stats['fwhm'], 0.17790825 * u.AA)