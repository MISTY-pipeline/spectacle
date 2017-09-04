from collections import OrderedDict

import astropy.units as u
import peakutils
from astropy.modeling import Parameter, Fittable2DModel
from astropy.modeling.fitting import LevMarLSQFitter
from ..models.converters import WavelengthConvert, VelocityConvert
from ..models import *


class LineFinder(Fittable2DModel):
    inputs = ('x', 'y')
    outputs = ('y',)
    input_units_strict = True
    input_units_allow_dimensionless = True

    threshold = Parameter(default=0.5)
    min_distance = Parameter(default=30, min=0)
    width = Parameter(default=10, min=0)
    center = Parameter(default=0, fixed=True, unit=u.Unit('Angstrom'))

    input_units = {'x': u.Unit('km/s')}

    def __init__(self, model, *args, **kwargs):
        super(LineFinder, self).__init__(*args, **kwargs)
        self._model = model

    @property
    def input_units_equivalencies(self):
        return {'x': [
            (u.Unit('km/s'), u.Unit('Angstrom'),
             lambda x: WavelengthConvert(self.center)(x * u.Unit('km/s')),
             lambda x: VelocityConvert(self.center)(x * u.Unit('Angstrom')))
        ]}

    def evaluate(self, x, y, threshold, min_distance, width, center):
        # Astropy fitters strip models of their unit information. However, the
        # first iterate of a fitter includes the quantity arrays passed to the
        # call method. If the input array is a quantity, immediately store the
        # quantity unit as a reference for future iterations.
        if isinstance(x, u.Quantity):
            self.input_units = {'x': x.unit}

        x = u.Quantity(x, self.input_units['x'])
        min_distance = u.Quantity(min_distance, self.input_units['x'])
        width = u.Quantity(width, self.input_units['x'])

        model = self._model.copy()

        indexes = peakutils.indexes(y, thres=threshold, min_dist=min_distance.value)

        peaks_x = peakutils.interpolate(x.value, y, ind=indexes, width=int(width.value))
        print("Found {} peaks.".format(len(peaks_x)))

        for peak in peaks_x:
            peak = u.Quantity(peak, x.unit)
            peak = peak.to('Angstrom', equivalencies=self.input_units_equivalencies['x'])
            model.add_line(lambda_0=peak)

        fitter = LevMarLSQFitter()

        self._result_model = fitter(model, x, y)

        return self._result_model(x)

    def _parameter_units_for_data_units(self, input_units, output_units):
        return OrderedDict([('min_distance', input_units['x']),
                            ('width', input_units['x'])])

    @property
    def result_model(self):
        return self._result_model