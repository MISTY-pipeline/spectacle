from collections import OrderedDict

import logging
import astropy.units as u
import numpy as np
from astropy.constants import c
from astropy.modeling import Fittable1DModel, Parameter

from ..utils import wave_to_vel_equiv

__all__ = ['VelocityConvert', 'WavelengthConvert',
           'DispersionConvert', 'FluxConvert', 'FluxDecrementConvert']


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

    input_units = {'x': u.Unit('Angstrom')}

    center = Parameter(default=0, fixed=True, unit=u.Unit('Angstrom'))

    @staticmethod
    def evaluate(x, center):
        # ln_lambda = np.log(x) - np.log(center)
        # vel = (c.cgs * ln_lambda).to('km/s').value

        vel = (c.cgs * ((x - center) / x)).to('km/s')

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

    input_units = {'x': u.Unit('km/s')}

    center = Parameter(default=0, fixed=True, unit=u.Unit('Angstrom'))

    @staticmethod
    def evaluate(x, center):
        wav = center * (1 + x / c.cgs)

        return wav


class DispersionConvert(Fittable1DModel):
    inputs = ('x',)
    outputs = ('x',)

    input_units_strict = True

    center = Parameter(default=0, fixed=True, unit=u.Unit('Angstrom'))

    @property
    def input_units_equivalencies(self):
        return {'x': wave_to_vel_equiv(self.center)}

    def evaluate(self, x, center):
        with u.set_enabled_equivalencies(self.input_units_equivalencies['x']):
            if x.unit.physical_type == 'speed':
                return x.to('Angstrom')
            elif x.unit.physical_type == 'length':
                return x.to('km/s')

        logging.warning("Unrecognized input units '{}'.".format(x.unit))

        return x

    def _parameter_units_for_data_units(self, input_units, output_units):
        return OrderedDict()


class FluxConvert(Fittable1DModel):
    inputs = ('y',)
    outputs = ('y',)

    @staticmethod
    def evaluate(y):
        return np.exp(-y) - 1

    def _parameter_units_for_data_units(self, input_units, output_units):
        return OrderedDict()


class FluxDecrementConvert(Fittable1DModel):
    inputs = ('y',)
    outputs = ('y',)

    @staticmethod
    def evaluate(y):
        return 1 - np.exp(-y) - 1

    def _parameter_units_for_data_units(self, input_units, output_units):
        return OrderedDict()
