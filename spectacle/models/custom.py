from astropy.modeling.models import Scale, RedshiftScaleFactor
from astropy.modeling import Parameter, Fittable2DModel
import astropy.units as u
from collections import OrderedDict
import numpy as np

from ..core.region_finder import find_regions


class SmartScale(Scale):
    input_units_strict = True

    factor = Parameter(default=1, min=0, fixed=True)

    @property
    def input_units(self):
        return {'x': u.Unit('Angstrom')}

    @staticmethod
    def evaluate(x, factor):
        """One dimensional Scale model function"""
        if isinstance(factor, u.Quantity):
            return_unit = factor.unit
            factor = factor.value

            if isinstance(x, u.Quantity):
                return (x.value * factor) * return_unit
        else:
            return factor * x

    def _parameter_units_for_data_units(self, input_units, output_units):
        return OrderedDict([('factor', None)])


class Redshift(RedshiftScaleFactor):
    z = Parameter(default=0, min=0)

    @property
    def input_units(*args, **kwargs):
        return {'x': u.Unit('Angstrom')}

    def _parameter_units_for_data_units(self, input_units, output_units):
        return OrderedDict([('z', None)])


class Masker(Fittable2DModel):
    inputs = ('x', 'y')
    outputs = ('x', 'y')

    def __init__(self, continuum=None, *args, **kwargs):
        super(Masker, self).__init__(*args, **kwargs)

        self._continuum = continuum

    def evaluate(self, x, y, *args, **kwargs):
        continuum = self._continuum if self._continuum is not None else np.zeros(y.shape)
        reg = find_regions(y, continuum=continuum)

        mask = np.logical_or.reduce([(x > x[rl]) & (x <= x[rr]) for rl, rr in reg])

        return np.ma.array(x, mask=~mask), np.ma.array(y, mask=~mask)