import peakutils
from spectacle.modeling import *
from spectacle.core.spectrum import Spectrum1D

from astropy.modeling import Fittable2DModel


class LineFinder(Fittable2DModel):
    inputs = ('x', 'y')
    outputs = ('y',)

    input_units_strict = True
    input_units_allow_dimensionless = True

    center = Parameter(default=1216, min=0, unit=u.Unit('Angstrom'))
    threshold = Parameter(default=0.5)
    min_distance = Parameter(default=30, min=0)

    input_units = {'x': u.Unit('km/s')}

    @property
    def input_units_equivalencies(self):
        return {'x': [
            (u.Unit('km/s'), u.Unit('Angstrom'),
             lambda x: WavelengthConvert(self.center)(x * u.Unit('km/s')),
             lambda x: VelocityConvert(self.center)(x * u.Unit('Angstrom')))
        ]}

    def evaluate(self, x, y, center, threshold, min_distance):
        # Astropy fitters strip modeling of their unit information. However, the
        # first iterate of a fitter includes the quantity arrays passed to the
        # call method. If the input array is a quantity, immediately store the
        # quantity unit as a reference for future iterations.
        if isinstance(x, u.Quantity):
            self.input_units = {'x': x.unit}

        x = u.Quantity(x, self.input_units['x'])
        min_distance = u.Quantity(min_distance, self.input_units['x'])

        indexes = peakutils.indexes(y, thres=threshold,
                                    min_dist=min_distance.value)

        spectrum = Spectrum1D(center=self.center)

        for ind in indexes:
            peak = u.Quantity(x[ind], x.unit)
            peak = peak.to(
                'Angstrom', equivalencies=self.input_units_equivalencies['x'])

            spectrum.add_line(lambda_0=peak, column_density=1e14 *
                              u.Unit('1/cm2'), fixed={'delta_v': True})

        fitter = LevMarLSQFitter()
        self._result_model = fitter(spectrum.optical_depth, x, y, maxiter=1000)

        return self._result_model(x)

    def _parameter_units_for_data_units(self, input_units, output_units):
        return OrderedDict([('min_distance', input_units['x']),
                            ('width', input_units['x'])])

    @property
    def result_model(self):
        return self._result_model
