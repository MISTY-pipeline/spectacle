from astropy.modeling.models import Scale, RedshiftScaleFactor
from astropy.modeling import Parameter, Fittable2DModel
import astropy.units as u
from collections import OrderedDict
import numpy as np

from ..core.region_finder import find_regions
from ..models.profiles import TauProfile
from ..models.converters import WavelengthConvert, VelocityConvert


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
    input_units_strict = True

    center = Parameter(default=0, fixed=True, unit=u.Unit('Angstrom'))

    input_units = {'x': u.Unit('Angstrom')}

    @property
    def input_units_equivalencies(self):
        return {'x': [
            (u.Unit('km/s'), u.Unit('Angstrom'),
             lambda x: WavelengthConvert(self.center)(x * u.Unit('km/s')),
             lambda x: VelocityConvert(self.center)(x * u.Unit('Angstrom')))
        ]}

    def __init__(self, continuum=None, line_list=None, rel_tol=1e-2, abs_tol=1e-4,
                 *args, **kwargs):
        super(Masker, self).__init__(*args, **kwargs)
        self._line_list = line_list

        # In the case where the user has provided a list of ion names, attempt
        # to find the ions in the database
        if self._line_list:
            self._line_list = [TauProfile(name=x) for x in self._line_list]

        self._continuum = continuum
        self._rel_tol = rel_tol
        self._abs_tol = abs_tol

    def evaluate(self, x, y, center):
        x = x.to('Angstrom', equivalencies=self.input_units_equivalencies['x'])

        continuum = self._continuum if self._continuum is not None else np.zeros(y.shape)

        reg = find_regions(y, continuum=continuum, rel_tol=self._rel_tol,
                           abs_tol=self._abs_tol)

        if self._line_list is not None:
            filt_reg = []

            for rl, rr in reg:
                print([print(x[rl], prof.lambda_0, x[rr]) for prof in self._line_list])
                if any([x[rl] <= prof.lambda_0 <= x[rr] for prof in self._line_list]):
                    filt_reg.append((rl, rr))

            reg = filt_reg

        mask = np.logical_or.reduce([(x > x[rl]) & (x <= x[rr]) for rl, rr in reg])

        return np.ma.array(x, mask=~mask), np.ma.array(y, mask=~mask)