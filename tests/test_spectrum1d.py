from spectacle.core.spectra import Spectrum1D
import numpy as np
from astropy.nddata import StdDevUncertainty
from astropy.units import Unit
import os


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

    def test_create_from_file(self):
        from astropy.io import fits

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

        hdulist = fits.open(os.path.join(path, "test_data.fits"))

        # The misty file reader returns a list of spectra
        specs = Spectrum1D.read(os.path.join(path, "test_data.fits"),
                                format='misty')

        assert all(isinstance(spec, Spectrum1D) for spec in specs)
        assert specs[0].data == hdulist[2].data['flux']

    def test_velocity(self):
        data = np.random.sample(100)

        spec = Spectrum1D(data)

        assert spec.velocity(50)[50] == 0