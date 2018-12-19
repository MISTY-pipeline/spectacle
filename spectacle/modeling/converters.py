import astropy.units as u
import numpy as np
from astropy.modeling import Fittable1DModel, Parameter

DOPPLER_CONVERT = {
    'optical': u.doppler_optical,
    'radio': u.doppler_radio,
    'relativistic': u.doppler_relativistic
}


class FluxConvert(Fittable1DModel):
    inputs = ('y',)
    outputs = ('y',)

    @staticmethod
    def evaluate(y):
        return np.exp(-y) - 1

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {}


class FluxDecrementConvert(Fittable1DModel):
    inputs = ('y',)
    outputs = ('y',)

    @staticmethod
    def evaluate(y):
        return 1 - np.exp(-y) - 1


class DispersionConvert(Fittable1DModel):
    """
    Convert dispersions into velocity space for use internally.

    Arguments
    ---------
    rest_wavelength : :class:`~astropy.units.Parameter`
        Wavelength for use in the equivalency conversions.
    """
    inputs = ('x',)
    outputs = ('x',)

    rest_wavelength = Parameter(default=0, unit=u.AA, fixed=True)

    input_units_allow_dimensionless = {'x': True}
    input_units = {'x': u.Unit('km/s')}

    linear = True
    fittable = True

    @property
    def input_units_equivalencies(self):
        return {'x': u.spectral() + u.doppler_optical(
            self.rest_wavelength.value * u.AA)}

    @staticmethod
    def evaluate(x, rest_wavelength):
        """One dimensional Scale model function"""
        disp_equiv = u.spectral() + u.doppler_optical(
            u.Quantity(rest_wavelength, u.AA))

        with u.set_enabled_equivalencies(disp_equiv):
            x = u.Quantity(x, u.Unit("km/s"))

        return x.value