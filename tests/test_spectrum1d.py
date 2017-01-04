from spectacle.core.spectra import Spectrum1D
import numpy as np
from astropy.nddata import StdDevUncertainty
from astropy.units import Unit


class TestSpectrum1D:
    def test_create_new(self):
        data = np.random.sample(100)
        spec = Spectrum1D(data)

        assert isinstance(spec, Spectrum1D)
        assert spec.data[0] == data[0]

    def test_create_with_dispersion(self):
        data = np.random.sample(100)
        dispersion = np.linspace(500, 600, 100)

        spec = Spectrum1D(data, dispersion=dispersion)

        assert isinstance(spec, Spectrum1D)
        assert spec.dispersion[0] == dispersion[0]

    def test_create_with_uncertainty(self):
        data = np.random.sample(100)
        uncert = np.sqrt(data)

        spec = Spectrum1D(data, uncertainty=uncert)

        assert isinstance(spec.uncertainty, StdDevUncertainty)
        assert spec.uncertainty.array[0] == uncert[0]

    def test_create_with_unit(self):
        data = np.random.sample(100)

        spec = Spectrum1D(data, unit="Jy", dispersion_unit="Angstrom")

        assert isinstance(spec.unit, Unit)
        assert isinstance(spec.dispersion_unit, Unit)
        assert spec.unit.name == "Jy"
        assert spec.dispersion_unit.name == "Angstrom"