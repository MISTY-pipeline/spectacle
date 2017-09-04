from astropy.modeling.models import Scale, RedshiftScaleFactor
import astropy.units as u
from collections import OrderedDict


class SmartScale(Scale):
    input_units_strict = True

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
        return OrderedDict([('factor', u.Unit('Angstrom'))])


class Redshift(RedshiftScaleFactor):
    @property
    def input_units(*args, **kwargs):
        return {'x': u.Unit('Angstrom')}

    def _parameter_units_for_data_units(self, input_units, output_units):
        return OrderedDict([('z', None)])