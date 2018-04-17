import logging
from collections import OrderedDict

import astropy.units as u
import numpy as np
import scipy.integrate as integrate
from astropy.constants import c, m_e
from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.fitting import LevMarLSQFitter

from ..core.region_finder import find_regions
from ..core.spectrum import Spectrum1D
from ..io.registries import line_registry
from ..modeling import *
from ..modeling.custom import Linear
from ..modeling.fitters import MCMCFitter
from ..utils import peak_finder, wave_to_vel_equiv
from .initializers import Voigt1DInitializer

# (np.pi * np.exp(2)) / (m_e.cgs * c.cgs) * 0.001
AMP_CONST = np.pi * np.e ** 2 / (
    m_e.cgs * c.cgs)  # 8.854187817e-13 * u.Unit('1/cm')
PROTON_CHARGE = u.Quantity(4.8032056e-10, 'esu')
TAU_FACTOR = ((np.sqrt(np.pi) * PROTON_CHARGE ** 2 /
               (m_e.cgs * c.cgs))).cgs
SIGMA = (np.pi * np.e ** 2) / (m_e * c.cgs ** 2)


class LineFinder(Fittable2DModel):
    """
    Line finder model.

    Parameters
    ----------
    center : float
        The lambda value used to convert from wavelength to velocity space.
    z : float
        Redshift value of the resultant spectrum.
    threshold : float
        Normalized threshold used in peak finder. Only the peaks with
        amplitude higher than the threshold will be detected.
    min_distance : float
        Minimum distance between each detected peak used in peak finder. The
        peak with the highest amplitude is preferred to satisfy this
        constraint.
    """
    inputs = ('x', 'y')
    outputs = ('y',)

    input_units_strict = True

    center = Parameter(default=0, min=0, unit=u.Unit('Angstrom'), fixed=True)
    redshift = Parameter(default=0, min=0, fixed=True)
    threshold = Parameter(default=0.1, min=0)
    min_distance = Parameter(default=2, min=0.1)
    width = Parameter(default=15, min=2)

    @property
    def input_units(self):
        return {'x': u.Unit('Angstrom')}

    def __init__(self, ion_name=None, data_type='optical_depth',
                 defaults=None, max_iter=2000, *args, **kwargs):
        super(LineFinder, self).__init__(*args, **kwargs)

        if data_type not in ('optical_depth', 'flux', 'flux_decrement'):
            logging.error("No available data type named '%s'. Defaulting to"
                          "'optical_depth'.", data_type)
            self._data_type = 'optical_depth'
        else:
            self._data_type = data_type

        self._result_model = None
        self._regions = None
        self._line_defaults = defaults or {}

        if 'lambda_0' in self._line_defaults:
            self.center = self._line_defaults.get('lambda_0')
            self._ion_info = line_registry.with_lambda(self.center)
        elif ion_name is not None:
            self._ion_info = line_registry.with_name(ion_name)
            self.center = self._ion_info['wave'] * line_registry['wave'].unit

        self.max_iter = max_iter

    @property
    def input_units_equivalencies(self):
        return {'x': wave_to_vel_equiv(self.center)}

    def __call__(self, x, *args, **kwargs):
        if isinstance(x, u.Quantity):
            self.input_units['x'] = x.unit
        else:
            logging.warning("Input 'x' is not a quantity.")

        super(LineFinder, self).__call__(x, *args, **kwargs)

        return self._result_model

    def evaluate(self, x, y, center, redshift, threshold, min_distance, width):
        """
        Evaluate `LineFinder` model.
        """
        # Units are stripped in the evaluate methods of models
        x = u.Quantity(x, unit=self.input_units['x'])
        logging.info("Input units: {}".format(x.unit))

        # Convert the min_distance from dispersion units to data elements
        min_ind = (np.abs(x.value - (x[0].value + min_distance))).argmin()

        logging.info("Min distance: %i elements.", min_ind)

        # Given each minima in the finder, construct a new spectrum object with
        # absorption lines at the given indicies.
        spectrum = Spectrum1D(center=self.center.value * self.center.unit,
                              redshift=self.redshift.value,
                              continuum=Linear(slope=u.Quantity(0, 1 / x.unit),
                                               intercept=(
                                                    0 if self._data_type == 'optical_depth' else 1) * u.Unit(""),
                                               fixed={'slope': True, 'intercept': True}))

        fit_spec_mod = self.generate_model(spectrum, x, y, np.max(y) - y if self._data_type == 'flux' else y, threshold, min_ind)
        prev_num_sm = -1

        # while len(fit_spec_mod._submodels) != prev_num_sm:
        #     prev_num_sm = len(fit_spec_mod._submodels)

        #     fit_spec_mod = self.generate(spectrum, x, y,
        #                                  np.abs(fit_spec_mod(x) - y).value,
        #                                  threshold, min_ind)

        #     import matplotlib.pyplot as plt

        #     f, ax = plt.subplots()

        #     ax.plot(x, y)
        #     ax.plot(x, fit_spec_mod(x))

        #     plt.show()

        #     print(prev_num_sm, len(fit_spec_mod._submodels))

        # Remove any extraneous lines that do not contain any useful info
        # fit_spec_mod, any_removed = remove_empty_lines(fit_spec_mod)

        logging.debug("End fitting.")

        self._result_model = spectrum

        # Calculate the regions in the raw data
        regions = find_regions(y, rel_tol=1e-2, abs_tol=1e-4,
                               continuum=spectrum._continuum_model(x).value)

        # Convert regions to a dictionary where the keys are the tuple of the
        # indicies
        reg_dict = {(reg[0], reg[1]): [] for reg in regions}

        # Add this line to the region dictionary
        for line in spectrum.line_models:
            redshifted_peak = spectrum._redshift_model(line.lambda_0 +
                                                       line.delta_lambda)

            for k in reg_dict.keys():
                mn, mx = x[k[0]], x[k[1]]

                if mn <= redshifted_peak <= mx:
                    reg_dict[k].append(line)

        # Set the region dictionary on the spectrum model object
        spectrum.regions = reg_dict

        logging.info("Found %i absorption regions.", len(reg_dict))

        return getattr(self._result_model, self._data_type)(x)

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return OrderedDict()

    def generate_model(self, spectrum, x, y, y_peaks, threshold, min_ind):
        # Take a first iteration of the minima finder
        indicies = peak_finder.indexes(
            y_peaks,
            thres=threshold,
            min_dist=min_ind,
            min_thresh=0 if self._data_type != 'optical_depth' else None,
            max_thresh=1 if self._data_type != 'optical_depth' else None)

        # Enhance the peak detection using interpolation
        # peaks = peak_finder.interpolate(x.value,
        #                                 np.max(y) - y if self._data_type == 'flux' else y,
        #                                 ind=indicies, width=min_ind)

        logging.info("Found %i minima.", len(indicies))

        # Generation spectrum model
        self.compose_lines(spectrum, indicies, x, y, threshold, min_ind)

        logging.debug("Begin fitting...")

        # Attempt to fit this new spectrum object to the data
        fitter = LevMarLSQFitter()

        fit_spec_mod = fitter(
            getattr(spectrum, self._data_type), x, y,
            maxiter=self.max_iter
            )

        # fit_spec_mod = getattr(spectrum, self._data_type)

        fitter = MCMCFitter() # LevMarLSQFitter()

        fit_spec_mod = fitter(
            fit_spec_mod, x, y,
            # maxiter=self.max_iter
            nwalkers=100, steps=500
            )

        # Update spectrum line model parameters with fitted results
        fit_line_mods = [smod for smod in fit_spec_mod if hasattr(smod, 'lambda_0')]

        if len(fit_line_mods) > 0:
            spectrum._line_model = np.sum(fit_line_mods)

        return fit_spec_mod

    def compose_lines(self, spectrum, indicies, x, y, threshold, min_ind):
        center = self.center.value * self.center.unit

        for ind in indicies:
            redshifted_peak = x[ind]
            deredshifted_peak = spectrum._redshift_model.inverse(
                redshifted_peak)

            # Get the index of this line model's centroid
            ind = (np.abs(x.value - redshifted_peak.value)).argmin()

            # Construct a dictionary with all of the true values for this
            # absorption line.
            line_kwargs = dict(lambda_0=center)

            # Based on the dispersion type, calculate the lambda or velocity
            # deltas.
            with u.set_enabled_equivalencies(self.input_units_equivalencies['x']):
                if x.unit.physical_type == 'length':
                    line_kwargs['delta_lambda'] = deredshifted_peak.to('Angstrom') - center.to(
                        'Angstrom')
                    line_kwargs['fixed'] = {
                        'delta_lambda': False,
                        'delta_v': True
                    }
                elif x.unit.physical_type == 'speed':
                    line_kwargs['delta_v'] = deredshifted_peak.to(
                        'km/s') - center.to('km/s')
                    line_kwargs['fixed'] = {
                        'delta_lambda': True,
                        'delta_v': False
                    }
                else:
                    raise ValueError("Could not get physical type of "
                                        "dispersion axis unit.")

                # Calculate some initial parameters for velocity and col density
                vel = x.to('km/s')

            line_kwargs.update(
                estimate_line_parameters(vel, y, ind, min_ind, center,
                    spectrum._continuum_model(x).value if self._data_type == 'flux' else None))

            line_kwargs.update(self._line_defaults)

            # Create a line profile model and add it to the spectrum object
            spectrum.add_line(**line_kwargs)


def estimate_line_parameters(x, y, ind, min_ind, center, continuum):
    bound_low = max(0, min(ind - int(min_ind * 0.5), x.size - 1))
    bound_up = max(0, min(ind + int(min_ind * 0.5), x.size - 1))
    mask = ((x > x[bound_low]) & (x < x[bound_up]))

    if continuum is not None:
        y = continuum - y

    mx = x[mask]
    my = y[mask]

    # Width can be estimated by the weighted 2nd moment of the x coordinate
    dx = mx - np.mean(mx)
    fwhm = 2 * np.sqrt(np.sum((dx * dx) * my) / np.sum(my))
    centroid = np.sum(mx * my) / np.sum(my)
    sigma = fwhm / 2.355

    # Amplitude is derived from area
    delta_x = mx[1:] - mx[:-1]
    sum_y = np.sum(my[1:] * delta_x)
    height = sum_y / (sigma * np.sqrt(2 * np.pi))

    from astropy.modeling.models import Gaussian1D
    from astropy.modeling.fitting import LevMarLSQFitter

    g = Gaussian1D(amplitude=height, mean=centroid, stddev=sigma)
    g_fit = LevMarLSQFitter()(g, mx, my)

    new_dx = x - np.mean(mx)
    new_delta_x = x[1:] - x[:-1]
    new_y = g_fit(x)
    new_fwhm = 2 * np.sqrt(np.sum((new_dx * new_dx) * new_y) / np.sum(new_y))
    new_sum_y = np.sum(new_y[1:] * new_delta_x)

    # import matplotlib.pyplot as plt
    # f, ax = plt.subplots()

    # ax.plot(x, y)
    # ax.plot(x, g(x))
    # ax.plot(x, g_fit(x))

    # Estimate the doppler b parameter
    v_dop = 0.60056120439322491 * new_fwhm

    # Estimate the column density
    f_value = line_registry.with_lambda(center)['osc_str']
    col_dens = (new_sum_y * u.Unit('kg/(cm * s * Angstrom)') * SIGMA * f_value * center).to('1/cm2')

    logging.info("""Estimated intial values:
    Column density: {:g}
    Doppler width: {:g}""".format(col_dens, v_dop.to('cm/s')))

    return dict(v_doppler=v_dop.to('cm/s'), column_density=col_dens)


def remove_empty_lines(model):
    sm_inds = [range(len(model._submodels))]

    for i in [x for x in sm_inds]:
        sm = model[i]

        if hasattr(sm, 'lambda_0'):
            if sm.equivalent_width() > sm.v_dopper:
                sm_inds.remove(i)

    new_mod = [x for i, x in enumerate(model) if i in sm_inds]

    return np.sum(new_mod), len(sm_inds) < len(model._submodels)