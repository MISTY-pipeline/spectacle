import astropy.units as u
import numpy as np
from spectacle.spectrumnd import SpectrumND


def test_init_basic():
    spec1d = SpectrumND(np.arange(10) * u.Angstrom,
                        np.random.sample(10) * u.Unit('erg/(s cm2 Angstrom)'))

    assert spec1d.spectral_axis.unit == u.Angstrom
    assert spec1d.flux.unit == u.Unit('erg/s/cm2/Angstrom')


def test_optical_depth():
    spec = SpectrumND(np.arange(10) * u.Angstrom,
                      np.random.sample(10) * u.Unit('erg/(s cm2 Angstrom)'))

    y = spec.optical_depth(3 * u.Angstrom)
    print(y)
    assert y is not False
