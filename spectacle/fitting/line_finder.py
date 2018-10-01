from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.models import Linear1D
from peakutils import indexes as find_indexes
import astropy.units as u
from astropy.modeling.fitting import LevMarLSQFitter
import logging

from ..modeling import Spectral1D, OpticalDepth1D
from ..modeling.models import _strip_units

class LineFinder1D(Fittable1DModel):
    inputs = ('x',)
    outputs = ('x',)

    threshold = Parameter(default=0.3, min=0, max=1)
    min_distance = Parameter(default=1, min=1)

    def __init__(self, defaults=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._y = None
        self._defaults = defaults or {}
        self._model_result = None

    def __call__(self, x, y, continuum=None, *args, **kwargs):
        self._y = y
        self._continuum = continuum

        super().__call__(x, *args, **kwargs)

        return self._model_result

    def evaluate(self, x, theshold, min_distance, *args, **kwargs):
        with u.set_enabled_equivalencies(u.spectral() + u.doppler_relativistic(1216 * u.AA)):
            x = u.Quantity(x, 'km/s')

        # Find peaks
        indexes = find_indexes(self._y, theshold, min_distance)

        lines = []

        for index in indexes:
            line_kwargs = self._defaults.copy()
            line_kwargs.setdefault('delta_v', x[index])

            logging.info("Found line at %s.", x[index])

            line = OpticalDepth1D(**line_kwargs)
            lines.append(line)

        logging.info("Found %s possible lines (theshold=%s, min_distance=%s).",
                     len(lines), theshold, min_distance)

        spec_mod = Spectral1D(lines, continuum=self._continuum)

        # fitter = LevMarLSQFitter()
        fit_spec_mod = spec_mod.fit_to(x, self._y)

        self._model_result = fit_spec_mod

        return fit_spec_mod(x)