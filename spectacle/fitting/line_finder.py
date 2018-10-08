import logging

import astropy.units as u
import numpy as np
from astropy.constants import c, m_e
from astropy.modeling import Fittable1DModel, Parameter

from ..modeling import OpticalDepth1D, Spectral1D
from ..utils.detection import region_bounds
from ..registries import line_registry

PROTON_CHARGE = u.Quantity(4.8032056e-10, 'esu')
TAU_FACTOR = ((np.sqrt(np.pi) * PROTON_CHARGE ** 2 /
               (m_e.cgs * c.cgs))).cgs


class LineFinder1D(Fittable1DModel):
    inputs = ('x',)
    outputs = ('x',)

    threshold = Parameter(default=0.3, min=0, max=1)
    min_distance = Parameter(default=1, min=1)

    def __init__(self, defaults=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._y = None
        self._continuum = None
        self._defaults = defaults or {}
        self._model_result = None

    def __call__(self, x, y, continuum=None, *args, **kwargs):
        self._y = y
        self._continuum = continuum

        super().__call__(x, *args, **kwargs)

        return self._model_result

    def evaluate(self, x, threshold, min_distance, *args, **kwargs):
        with u.set_enabled_equivalencies(u.spectral() + u.doppler_relativistic(1216 * u.AA)):
            x = u.Quantity(x, 'km/s')

        # Find peaks
        bounds = region_bounds(self._y, x, distance=min_distance, height=threshold)

        lines = []

        for mn_bnd, mx_bnd in [x for x in bounds]:
            line_kwargs = self._defaults.copy()

            # Calculate the centroid of this region
            centroid = x[mn_bnd + int((mx_bnd - mn_bnd) * 0.5)]

            # Check that the range encompassed by the bounds is reasonably
            if mx_bnd - mn_bnd < 3:
                logging.warning("Bounds encompassing feature at %s do not "
                                "provide enough data; ignoring feature. "
                                "(Data points: %i).",
                                centroid, mx_bnd - mn_bnd)
                bounds.remove((mn_bnd, mx_bnd))
                continue

            # Estimate the doppler b and column densities for this line
            centroid, v_dop, col_dens = parameter_estimator(
                (x[mn_bnd], x[mx_bnd]), x, self._y)

            line_kwargs.setdefault('delta_v', centroid)
            line_kwargs.setdefault('v_doppler', v_dop)
            line_kwargs.setdefault('column_density', col_dens)
            line_kwargs.setdefault('bounds',
                                   {'delta_v': (x[mn_bnd].value, x[mx_bnd].value)})

            line = OpticalDepth1D(**line_kwargs)
            lines.append(line)

        logging.info("Found %s possible lines (theshold=%s, min_distance=%s).",
                     len(lines), threshold, min_distance)

        spec_mod = Spectral1D(lines, continuum=self._continuum)

        # fitter = LevMarLSQFitter()
        fit_spec_mod = spec_mod.fit_to(x, self._y)
        fit_spec_mod.line_bounds = bounds

        self._model_result = fit_spec_mod

        return fit_spec_mod(x)


def parameter_estimator(bounds, x, y):
    bound_low, bound_up = bounds
    mid_diff = (bound_up - bound_low) * 0.15  # Expand the regions a little
    mask = ((x >= (bound_low - mid_diff)) & (x <= (bound_up + mid_diff)))

    mx, my = x[mask], y[mask]

    # Width can be estimated by the weighted 2nd moment of the x coordinate
    dx = mx - np.mean(mx)
    fwhm = 2 * np.sqrt(np.sum((dx * dx) * my) / np.sum(my))
    center = np.sum(mx * my) / np.sum(my)
    sigma = fwhm / 2.355

    # Amplitude is derived from area
    delta_x = mx[1:] - mx[:-1]
    sum_y = np.sum(my[1:] * delta_x)
    height = sum_y / (sigma * np.sqrt(2 * np.pi))

    # Estimate the doppler b parameter
    v_dop = (2 * fwhm).to('km/s')

    # Estimate the column density
    col_dens = (sum_y.value * v_dop.value * 1e10) * u.Unit('1/cm2')
    col_dens = np.log10(col_dens.value)

    logging.info("""Estimated initial values:
    Centroid: {:g}
    Column density: {:g}
    Doppler width: {:g}""".format(center, col_dens, v_dop))

    return center, v_dop, col_dens