from astropy import units as u
import numpy as np
from spectacle.modeling import Spectral1D, OpticalDepth1D
from ..modeling.lsfs import COSLSFModel, GaussianLSFModel, LSFModel
from astropy.convolution import Box1DKernel


def test_output_data():
    pass


def test_cos_lsf():
    line1 = OpticalDepth1D("HI1216", v_doppler=500 * u.km/u.s, column_density=14)
    line2 = OpticalDepth1D("OVI1038", v_doppler=500 * u.km/u.s, column_density=15)

    x = np.linspace(1000, 1300, 1000) * u.Unit('Angstrom')

    spec_mod_with_lsf_in_init = Spectral1D([line1, line2], continuum=1, lsf='cos', output='flux')

    spec_mod = Spectral1D([line1, line2], continuum=1, output='flux')
    spec_mod_with_lsf = spec_mod.with_lsf('cos')

    assert len(x) == len(spec_mod_with_lsf_in_init(x))
    assert isinstance(spec_mod_with_lsf_in_init.lsf_kernel, COSLSFModel)
    assert len(x) == len(spec_mod_with_lsf(x))
    assert isinstance(spec_mod_with_lsf.lsf_kernel, COSLSFModel)


def test_gaussian_lsf():
    line1 = OpticalDepth1D("HI1216", v_doppler=500 * u.km / u.s,
                           column_density=14)
    line2 = OpticalDepth1D("OVI1038", v_doppler=500 * u.km / u.s,
                           column_density=15)

    x = np.linspace(1000, 1300, 1000) * u.Unit('Angstrom')

    spec_mod_with_lsf_in_init = Spectral1D([line1, line2], continuum=1,
                                           lsf='gaussian', output='flux')

    spec_mod = Spectral1D([line1, line2], continuum=1, output='flux')
    spec_mod_with_lsf = spec_mod.with_lsf('gaussian')

    assert len(x) == len(spec_mod_with_lsf_in_init(x))
    assert isinstance(spec_mod_with_lsf_in_init.lsf_kernel, GaussianLSFModel)
    assert len(x) == len(spec_mod_with_lsf(x))
    assert isinstance(spec_mod_with_lsf.lsf_kernel, GaussianLSFModel)


def test_custom_kernel_lsf():
    from astropy.modeling.models import Chebyshev1D
    kernel = Box1DKernel(width=10)

    line1 = OpticalDepth1D("HI1216", v_doppler=500 * u.km / u.s,
                           column_density=14)
    line2 = OpticalDepth1D("OVI1038", v_doppler=500 * u.km / u.s,
                           column_density=15)

    x = np.linspace(1000, 1300, 1000) * u.Unit('Angstrom')

    spec_mod_with_lsf_in_init = Spectral1D([line1, line2], continuum=1,
                                           lsf=kernel, output='flux')

    spec_mod = Spectral1D([line1, line2], continuum=1, output='flux')
    spec_mod_with_lsf = spec_mod.with_lsf(kernel)

    assert len(x) == len(spec_mod_with_lsf_in_init(x))
    assert isinstance(spec_mod_with_lsf_in_init.lsf_kernel, LSFModel)
    assert len(x) == len(spec_mod_with_lsf(x))
    assert isinstance(spec_mod_with_lsf.lsf_kernel, LSFModel)


def test_dispersion_convert():
    line1 = OpticalDepth1D("HI1216", v_doppler=500 * u.km / u.s,
                           column_density=14)
    line2 = OpticalDepth1D("OVI1038", v_doppler=500 * u.km / u.s,
                           column_density=15)

