import peakutils
from astropy.modeling import Parameter, Fittable2DModel
from ..models import *
from ..core.spectrum import Spectrum1D

import logging

from ..utils.utils import find_bounds


class LineFinder(Fittable2DModel):
    inputs = ('x', 'y')
    outputs = ('y',)
    input_units_strict = True
    input_units_allow_dimensionless = True

    threshold = Parameter(default=0.5)
    min_distance = Parameter(default=30, min=0)

    input_units = {'x': u.Unit('km/s')}

    @property
    def input_units_equivalencies(self):
        return {'x': [
            (u.Unit('km/s'), u.Unit('Angstrom'),
             lambda x: WavelengthConvert(self._base_model.center_0)(x * u.Unit('km/s')),
             lambda x: VelocityConvert(self._base_model.center_0)(x * u.Unit('Angstrom')))
        ]}

    def evaluate(self, x, y, threshold, min_distance):
        # Astropy fitters strip models of their unit information. However, the
        # first iterate of a fitter includes the quantity arrays passed to the
        # call method. If the input array is a quantity, immediately store the
        # quantity unit as a reference for future iterations.
        if isinstance(x, u.Quantity):
            self.input_units = {'x': x.unit}

        x = u.Quantity(x, self.input_units['x'])
        min_distance = u.Quantity(min_distance, self.input_units['x'])

        indexes = peakutils.indexes(y, thres=threshold, min_dist=min_distance.value)

        spectrum = Spectrum1D(center=self._center)

        lines = []

        for ind in indexes:
            peak = u.Quantity(x[ind], x.unit)
            peak = peak.to('Angstrom', equivalencies=self.input_units_equivalencies['x'])

            spectrum.add_line(lambda_0=peak)

        fitter = LevMarLSQFitter()
        self._result_model = fitter(spectrum.tau, x, y)

        return self._result_model(x)

    def _parameter_units_for_data_units(self, input_units, output_units):
        return OrderedDict([('min_distance', input_units['x']),
                            ('width', input_units['x'])])

    @property
    def result_model(self):
        return self._result_model