from astropy.modeling import Fittable1DModel, Parameter
import astropy.units as u
from astropy.constants import c
from collections import OrderedDict
import numpy as np


class VelocityConvert(Fittable1DModel):
    """
    Model to convert from wavelength space to velocity space.

    Parameters
    ----------
    center : float
        Central wavelength.

    Notes
    -----
    Model formula:

        .. math:: v = \frac{\lambda - \lambda_c}{\lambda} c
    """
    inputs = ('x',)
    outputs = ('x',)
    input_units_strict = True

    center = Parameter(default=0, fixed=True, unit=u.Unit('Angstrom'))

    @property
    def input_units(self, *args, **kwargs):
        return {'x': u.Unit('Angstrom')}

    @staticmethod
    def evaluate(x, center):
        # ln_lambda = np.log(x) - np.log(center)
        # vel = (c.cgs * ln_lambda).to('km/s').value

        vel = (c.cgs * ((x - center)/x)).to('km/s')

        return vel


class WavelengthConvert(Fittable1DModel):
    """
    Model to convert from velocity space to wavelength space.

    Parameters
    ----------
    center : float
        Central wavelength.

    Notes
    -----
    Model formula:

        .. math:: \lambda = \lambda_c (1 + \frac{\lambda}{c}
    """
    inputs = ('x',)
    outputs = ('x',)
    input_units_strict = True

    center = Parameter(default=0, fixed=True, unit=u.Unit('Angstrom'))

    @property
    def input_units(self, *args, **kwargs):
        return {'x': u.Unit('km/s')}

    @staticmethod
    def evaluate(x, center):
        wav = center * (1 + x / c.cgs)

        return wav


class DispersionConvert(Fittable1DModel):
    inputs = ('x',)
    outputs = ('x',)
    input_units_strict = True
    input_units_allow_dimensionless = True

    center = Parameter(default=0, fixed=True, unit=u.Unit('Angstrom'))

    input_units = {'x': u.Unit('km/s')}

    @property
    def input_units_equivalencies(self):
        return {'x': [
            (u.Unit('km/s'), u.Unit('Angstrom'),
             lambda x: WavelengthConvert(self.center)(x * u.Unit('km/s')),
             lambda x: VelocityConvert(self.center)(x * u.Unit('Angstrom')))
        ]}

    def evaluate(self, x, center, *args, **kwargs):
        # Astropy fitters strip models of their unit information. However, the
        # first iterate of a fitter includes the quantity arrays passed to the
        # call method. If the input array is a quantity, immediately store the
        # quantity unit as a reference for future iterations.
        if isinstance(x, u.Quantity):
            self.input_units = {'x': x.unit}

        x = u.Quantity(x, self.input_units['x'])

        return x.to('Angstrom', equivalencies=self.input_units_equivalencies['x'])

    def _parameter_units_for_data_units(self, input_units, output_units):
        return OrderedDict([('center', u.Unit('Angstrom'))])


class FluxConvert(Fittable1DModel):
    inputs = ('y',)
    outputs = ('y',)

    @staticmethod
    def evaluate(y):
        return np.exp(-y) - 1


class FluxDecrementConvert(Fittable1DModel):
    inputs = ('y',)
    outputs = ('y',)

    @staticmethod
    def evaluate(y):
        return 1 - np.exp(-y) - 1
