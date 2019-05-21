import logging

import astropy.units as u
import numpy as np
from astropy.constants import c, m_e
from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.fitting import LevMarLSQFitter

from ..modeling import OpticalDepth1D, Spectral1D
from ..utils.detection import region_bounds
from ..utils.misc import DOPPLER_CONVERT
from ..registries import line_registry

PROTON_CHARGE = u.Quantity(4.8032056e-10, 'esu')
TAU_FACTOR = (np.pi * PROTON_CHARGE ** 2 /
               (m_e.cgs * c.cgs)).cgs


class LineFinder1D(Fittable2DModel):
    inputs = ('x', 'y')
    outputs = ('y',)

    @property
    def input_units_allow_dimensionless(self):
        return {'x': False, 'y': True}

    threshold = Parameter(default=0, fixed=True)
    min_distance = Parameter(default=10.0, min=1, fixed=True)

    def __init__(self, ions=None, continuum=None, defaults=None, z=None,
                 auto_fit=True, velocity_convention='relativistic',
                 output='flux', fitter=None, with_rejection=False,
                 fitter_args=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._ions = ions or []
        self._continuum = continuum
        self._defaults = defaults or {}
        self._z = z or 0
        self._model_result = None
        self._auto_fit = auto_fit
        self._output = output
        self._velocity_convention = velocity_convention
        self._fitter_args = fitter_args or {}
        self._fitter = fitter or LevMarLSQFitter()
        self._with_rejection = with_rejection

    @property
    def model_result(self):
        return self._model_result

    def __call__(self, x, *args, auto_fit=None, **kwargs):
        if auto_fit is not None:
            self._auto_fit = auto_fit

        if x.unit.physical_type == 'speed' and len(self._ions) != 1:
            raise ReferenceError("The line finder will not be able to parse "
                                 "ion information in velocity space without "
                                 "being given explicit ion reference in the "
                                 "defaults dictionary.")

        super().__call__(x, *args, **kwargs)

        return self._model_result

    def evaluate(self, x, y, threshold, min_distance, *args, **kwargs):
        spec_mod = Spectral1D(continuum=self._continuum, output=self._output)

        # Generate the subset of the table for the ions chosen by the user
        sub_registry = line_registry

        if len(self._ions) > 0:
            # In this case, the user has provided a list of ions for their
            # spectrum. Create a subset of the line registry so that only
            # these ions will be searched when attempting to identify.
            sub_registry = line_registry.subset(self._ions)

        # Convert the min_distance from dispersion units to data elements.
        # Assumes uniform spacing.
        # min_ind = (np.abs(x.value - (x[0].value + min_distance))).argmin()

        # Find peaks
        regions = region_bounds(x, y, threshold=threshold,
                                min_distance=min_distance)
        lines = []

        for centroid, (mn_bnd, mx_bnd), is_absorption, buried in regions.values():
            mn_bnd, mx_bnd = mn_bnd * x.unit, mx_bnd * x.unit
            sub_x, vel_mn_bnd, vel_mx_bnd = None, None, None

            line_kwargs = {}

            # For the case where the user has provided a list of ions with a
            # dispersion in wavelength or frequency, convert each ion to
            # velocity space individually to avoid making assumptions of their
            # kinematics.
            if x.unit.physical_type in ('length', 'frequency'):
                line = sub_registry.with_lambda(centroid)

                disp_equiv = u.spectral() + DOPPLER_CONVERT[
                    self._velocity_convention](line['wave'])

                with u.set_enabled_equivalencies(disp_equiv):
                    sub_x = u.Quantity(x, 'km/s')
                    vel_mn_bnd, vel_mx_bnd, vel_centroid = mn_bnd.to('km/s'), \
                                                           mx_bnd.to('km/s'), \
                                                           centroid.to('km/s')
            else:
                line = sub_registry.with_name(self._ions[0])

            line_kwargs.update({
                'name': line['name'],
                'lambda_0': line['wave'],
                'gamma': line['gamma'],
                'f_value': line['osc_str']})

            # Estimate the doppler b and column densities for this line.
            # For the parameter estimator to be accurate, the spectrum must be
            # continuum subtracted.
            v_dop, col_dens, nmn_bnd, nmx_bnd = parameter_estimator(
                centroid=centroid,
                bounds=(vel_mn_bnd or mn_bnd, vel_mx_bnd or mx_bnd),
                x=sub_x or x,
                y=spec_mod.continuum(sub_x or x) - y if is_absorption else y,
                ion_info=line_kwargs,
                buried=buried)

            if np.isinf(col_dens) or np.isnan(col_dens):
                continue

            estimate_kwargs = {
                'v_doppler': v_dop,
                'column_density': col_dens,
                'fixed': {},
                'bounds': {},
            }

            # Depending on the dispersion unit information, decide whether
            # the fitter should consider delta values in velocity or
            # wavelength/frequency space.
            if x.unit.physical_type in ('length', 'frequency'):
                estimate_kwargs['delta_lambda'] = centroid - line['wave']
                estimate_kwargs['fixed'].update({'delta_v': True})
                # TODO: enable bounds checking on lambda values
                # estimate_kwargs['bounds'].update({
                #     'delta_lambda': (mn_bnd.value - centroid.value,
                #                      mx_bnd.value - centroid.value)})
            else:
                # In velocity space, the centroid *should* be zero for any
                # line given that the rest wavelength is taken as its lamba_0
                # in conversions. Thus, the given centroid is akin to the
                # velocity offset.
                estimate_kwargs['delta_v'] = centroid
                estimate_kwargs['fixed'].update({'delta_lambda': True})
                estimate_kwargs['bounds'].update({
                    'delta_v': (mn_bnd.value, mx_bnd.value)})

            line_kwargs.update(estimate_kwargs)
            line_kwargs.update(self._defaults.copy())

            line = OpticalDepth1D(**line_kwargs)
            lines.append(line)

        logging.debug("Found %s possible lines (theshold=%s, min_distance=%s).",
                      len(lines), threshold, min_distance)

        if len(lines) == 0:
            return np.zeros(x.shape)

        spec_mod = Spectral1D(lines,
                              continuum=self._continuum,
                              output=self._output,
                              z=self._z)

        if self._auto_fit:
            if isinstance(self._fitter, LevMarLSQFitter):
                if 'maxiter' not in self._fitter_args:
                    self._fitter_args['maxiter'] = 1000

            fit_spec_mod = self._fitter(spec_mod, x, y, **self._fitter_args)
        else:
            fit_spec_mod = spec_mod

        # The parameter values on the underlying compound model must also be
        # updated given the new fitted parameters on the Spectral1D instance.
        # FIXME: when fitting without using line finder, these values will not
        # be updated in the compound model.
        for pn in fit_spec_mod.param_names:
            pv = getattr(fit_spec_mod, pn)
            setattr(fit_spec_mod._compound_model, pn, pv)

        fit_spec_mod.line_regions = regions

        self._model_result = fit_spec_mod

        return fit_spec_mod(x)


def parameter_estimator(centroid, bounds, x, y, ion_info, buried=False):
    # bound_low, bound_up = bounds
    # mid_diff = (bound_up - bound_low)
    # new_bound_low, new_bound_up = (bound_low - mid_diff), (bound_up + mid_diff)

    new_bound_low, new_bound_up = bounds
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

    # Estimate the column density
    # col_dens = (v_dop / TAU_FACTOR * c.cgs ** 2).to('1/cm2')
    col_dens = (sum_y / (TAU_FACTOR * ion_info['lambda_0'] * ion_info['f_value'])).to('1/cm2')
    ln_col_dens = np.log10(col_dens.value)

    if buried:
        ln_col_dens -= 0.1

    logging.info("""Estimated initial values:
    Ion: {}
    Centroid: {:g} ({:g})
    Column density: {:g}, ({:g})
    Doppler width: {:g}""".format(ion_info['name'], centroid,
                                  ion_info['lambda_0'], ln_col_dens,
                                  col_dens, v_dop))

    return v_dop, ln_col_dens, new_bound_low, new_bound_up