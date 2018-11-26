import logging

import astropy.units as u
import numpy as np
from astropy.constants import c, m_e
from astropy.modeling import Fittable1DModel, Parameter

from ..modeling import OpticalDepth1D, Spectral1D
from ..utils.detection import region_bounds
from ..registries import line_registry

PROTON_CHARGE = u.Quantity(4.8032056e-10, 'esu')
TAU_FACTOR = (np.pi * PROTON_CHARGE ** 2 /
               (m_e.cgs * c.cgs)).cgs


class LineFinder1D(Fittable1DModel):
    inputs = ('x',)
    outputs = ('y',)

    threshold = Parameter(default=0.1, min=0, max=1)
    min_distance = Parameter(default=10.0, min=1, max=100)

    def __init__(self, y, continuum=None, defaults=None, auto_fit=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._y = y
        self._continuum = continuum
        self._defaults = defaults or {}
        self._model_result = None
        self._auto_fit = auto_fit

    @property
    def model_result(self):
        return self._model_result

    def _frac_guess(self, value):
        flr = np.floor(value)
        cl = np.ceil(value)
        frac = np.modf(value)[0]

        return flr * (1 - frac) + cl * frac

    def __call__(self, *args, auto_fit=None, **kwargs):
        if auto_fit is not None:
            self._auto_fit = auto_fit

        return super().__call__(*args, **kwargs)

    def evaluate(self, x, threshold, min_distance, *args, **kwargs):
        with u.set_enabled_equivalencies(u.spectral() + u.doppler_relativistic(1216 * u.AA)):
            x = u.Quantity(x, 'km/s')

        # Convert the min_distance from dispersion units to data elements.
        # Assumes uniform spacing.
        min_ind = (np.abs(x.value - (x[0].value + min_distance))).argmin()

        # Find peaks
        regions = region_bounds(x, self._y)

        lines = []

        for (mn_bnd, mx_bnd), (centroid, buried) in [x for x in regions.items()]:
            line_kwargs = self._defaults.copy()
            mn_bnd, mx_bnd = mn_bnd * x.unit, mx_bnd * x.unit

            # Estimate the doppler b and column densities for this line
            v_dop, col_dens, nmn_bnd, nmx_bnd = parameter_estimator(
                centroid=centroid,
                bounds=(mn_bnd, mx_bnd),
                x=x,
                y=self._y,
                ion_name=line_kwargs.get('name'),
                buried=buried)

            estimate_kwargs = {
                'delta_v': centroid,
                'v_doppler': v_dop,
                'column_density': col_dens,
                'bounds': {
                    'delta_v': (mn_bnd.value, mx_bnd.value)
                               if not buried else (x.value[0], x.value[-1]),
                },
            }
            line_kwargs.update(estimate_kwargs)

            line = OpticalDepth1D(**line_kwargs)

            print(np.trapz(line(x), x))
            lines.append(line)

        logging.debug("Found %s possible lines (theshold=%s, min_distance=%s).",
                      len(lines), threshold, min_distance)

        if len(lines) == 0:
            return np.zeros(x.shape)

        spec_mod = Spectral1D(lines, continuum=self._continuum, output='optical_depth')

        # fitter = LevMarLSQFitter()
        if self._auto_fit:
            fit_spec_mod = spec_mod.fit_to(x, self._y, kwargs={'maxiter': 2000})
        else:
            fit_spec_mod = spec_mod

        fit_spec_mod.line_regions = regions

        self._model_result = fit_spec_mod

        return fit_spec_mod(x)


def parameter_estimator(centroid, bounds, x, y, ion_name, buried):
    bound_low, bound_up = bounds
    mid_diff = (bound_up - bound_low)

    if buried:
        mid_diff *= 2

    new_bound_low, new_bound_up =(bound_low - mid_diff), (bound_up + mid_diff)
    mask = ((x >= new_bound_low) & (x <= new_bound_up))
    mx, my = x[mask], y[mask]

    # Width can be estimated by the weighted 2nd moment of the x coordinate
    dx = mx - np.mean(mx)
    fwhm = 2 * np.sqrt(np.sum((dx * dx) * my) / np.sum(my))
    sigma = fwhm / 2.355

    # Amplitude is derived from area
    delta_x = mx[1:] - mx[:-1]
    sum_y = np.sum(my[1:] * delta_x)
    height = sum_y / (sigma * np.sqrt(2 * np.pi))

    # Estimate the doppler b parameter
    v_dop = (np.sqrt(2) * np.sqrt(np.pi) * sigma).to('km/s')

    # Get information about the ion
    ion = line_registry.with_name(ion_name)

    # Estimate the column density
    # col_dens = (v_dop / TAU_FACTOR * c.cgs ** 2).to('1/cm2')
    col_dens = (sum_y / (TAU_FACTOR * ion['wave'] * ion['osc_str'])).to('1/cm2')
    col_dens = np.log10(col_dens.value)

    logging.info("""Estimated initial values:
    Centroid: {:g}
    Column density: {:g}
    Doppler width: {:g}""".format(centroid, col_dens, v_dop))

    return v_dop, col_dens, new_bound_low, new_bound_up