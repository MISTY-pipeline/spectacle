import logging
from collections import OrderedDict

import astropy.units as u
import numpy as np
import peakutils
import scipy.integrate as integrate
from astropy.constants import c, m_e
from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.models import Linear1D
from astropy.modeling.fitting import LevMarLSQFitter

from ..core.spectrum import Spectrum1D
from ..modeling import *
from ..modeling.fitters import MCMCFitter
from ..io.registries import line_registry
from ..core.region_finder import find_regions

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
    input_units_allow_dimensionless = True

    center = Parameter(default=0, min=0, unit=u.Unit('Angstrom'), fixed=True)
    redshift = Parameter(default=0, min=0, fixed=True)
    threshold = Parameter(default=0.01, min=0)
    min_distance = Parameter(default=30, min=0.1)
    rel_tol = Parameter(default=1e-2, min=1e-10, max=1)
    abs_tol = Parameter(default=1e-4, min=1e-10, max=1)
    width = Parameter(default=15, min=2)

    input_units = {}

    def __init__(self, x, y, ion_name=None, data_type='optical_depth',
                 defaults=None,
                 *args, **kwargs):
        super(LineFinder, self).__init__(*args, **kwargs)

        if data_type not in ('optical_depth', 'flux', 'flux_decrement'):
            logging.error("No available data type named '%s'. Defaulting to"
                          "'optical_depth'.", data_type)
            self._data_type = 'optical_depth'
        else:
            self._data_type = data_type

        self._result_model = None
        self._estimated_model = None
        self._regions = None
        self._x = x
        self.input_units['x'] = x.unit
        self._y = y
        self._line_defaults = defaults or {}

        if 'lambda_0' in self._line_defaults:
            self.center = self._line_defaults.get('lambda_0')
            self._ion_info = line_registry.with_lambda(self.center)
        elif ion_name is not None:
            self._ion_info = line_registry.with_name(ion_name)
            self.center = self._ion_info['wave'] * line_registry['wave'].unit

    @property
    def input_units_equivalencies(self):
        """
        Unit equivalencies used by `LineFinder`.
        """
        vel_to_wave = (u.Unit('km/s'), u.Unit('Angstrom'),
                       lambda x: WavelengthConvert(
                           self.center)(x * u.Unit('km/s')),
                       lambda x: VelocityConvert(self.center)(
                           x * u.Unit('Angstrom')))

        u.set_enabled_equivalencies([vel_to_wave])

        return {'x': [vel_to_wave]}

    def evaluate(self, x, y, center, redshift, threshold, min_distance,
                 rel_tol, abs_tol, width):
        """
        Evaluate `LineFinder` model.
        """
        # Astropy fitters strip modeling of their unit information. However, the
        # first iterate of a fitter includes the quantity arrays passed to the
        # call method. If the input array is a quantity, immediately store the
        # quantity unit as a reference for future iterations.
        if not isinstance(x, u.Quantity):
            x = u.Quantity(x, self._x.unit)

        if self._data_type != 'optical_depth':
            y = max(y) - y

        indexes = peakutils.indexes(y, thres=threshold, min_dist=min_distance)

        abs_lines = Linear1D(slope=0,
                             intercept=0 if self._data_type == 'optical_depth' else 1,
                             fixed={'slope': True, 'intercept': True})

        # Get the interesting regions within the data provided. Don't bother
        # calculating more than once, since y doesn't change.
        if self._regions is None:
            self._regions = find_regions(y, rel_tol=rel_tol, abs_tol=abs_tol)

        for ind in indexes:
            peak = u.Quantity(x[ind], x.unit)
            # Given the peak found by the peak finder, select the corresponding
            # region found in the region finder
            # filt_reg = [(rl, rr) for rl, rr in self._regions
            #             if x[rl] <= peak <= x[rr]]

            # Create the mask that can be applied to the original data to produce
            # data only with the particular region we're interested in
            # mask = np.logical_or.reduce([(x > x[rl]) & (x <= x[rr])
            #                              for rl, rr in filt_reg])
            mask = ((x > x[ind - int(width)]) & (x < x[ind + int(width)]))

            # Estimate the voigt parameters
            voigt = ExtendedVoigt1D()
            initializer = Voigt1DInitializer()
            init_voigt = initializer.initialize(voigt,
                                                x[mask].value,
                                                y[mask])

            # Fit the spectral line to the voigt line we initialized above
            fitter = LevMarLSQFitter()

            fit_line = fitter(init_voigt, x[mask].value, y[mask], maxiter=200)

            abs_lines = abs_lines + fit_line

        # Fit the spectrum mode to the given data
        # fitter = LevMarLSQFitter()

        # fitter(spectrum.optical_depth, x, y, maxiter=1000)
        self._estimated_model = abs_lines

        return self._estimate_voigt()(x)

    def _parameter_units_for_data_units(self, input_units, output_units):
        return OrderedDict()

    @property
    def result_model(self):
        """
        Returns the resultant spectrum model that is constructed by the
        `LineFinder`.
        """
        if self._result_model is None:
            self._result_model = self._estimate_voigt()

        return self._result_model

    @property
    def regions(self):
        """
        Returns the list of tuples specifying the beginning and end indices of
        identified absorption/emission regions.
        """
        return self._regions

    def fit(self, *args, **kwargs):
        fitter = LevMarLSQFitter()  # MCMCFitter() #
        fit_finder = fitter(self, self._x, self._y, self._y)

        return fit_finder.result_model

    def _estimate_voigt(self):
        spectrum = Spectrum1D(center=self.center, redshift=self.redshift)

        # Iterate over the found lines only if they exist
        logging.debug("Model has %s submodels." % self._estimated_model.n_submodels())
        
        if self._estimated_model.n_submodels() > 1:
            for line in [x for x in self._estimated_model][1:]:
                center = self.center.value * self.center.unit
                redshifted_center = spectrum._redshift_model(center)
                line_kwargs = {'lambda_0': center}

                # Store the peak of this absorption feature, note that this is not
                # the center of the ion
                peak = line.x_0.value * self._x.unit
                deredshifted_peak = spectrum._redshift_model.inverse(peak)

                # Get the index of this line model's centroid
                ind = (np.abs(self._x.value - peak.value)).argmin()

                if self._x.unit.physical_type == 'length':
                    line_kwargs['delta_lambda'] = deredshifted_peak.to('Angstrom') - center.to(
                        'Angstrom')
                elif self._x.unit.physical_type == 'speed':
                    line_kwargs['delta_v'] = deredshifted_peak.to(
                        'km/s') - center.to('km/s')
                else:
                    logging.error(
                        "Could not get physical type of dispersion axis unit.")

                # Estimate the doppler b parameter
                fwhm_L = line.fwhm_L * self._x.unit
                fwhm = line.fwhm * self._x.unit

                # v_dop = (c.cgs * fwhm_L / (2 * fwhm * peak_lambda.value)).to('cm/s')
                # v_dop = (np.sqrt(
                #     1729929 * fwhm_L ** 2 - 26730000 * fwhm_L * fwhm + 25000000 * fwhm ** 2) / (
                #     5000 * np.sqrt(np.log(16))))
                v_dop = np.sqrt(0.0249576 * fwhm_L ** 2 - 0.385632 * fwhm_L *
                                fwhm + 0.360674 * fwhm ** 2)

                if self._x.unit.physical_type == 'length':
                    v_dop = (v_dop * c.cgs) / center

                # Estimate the column density
                f_value = self._line_defaults.get('f_value',
                                                self._ion_info['osc_str'])

                # col_dens = (self._y[ind] / (TAU_FACTOR * f_value * deredshifted_peak
                #                        * line(self._x[ind].value) * u.Unit('s/km'))).to(
                #     '1/cm2') * 5
                col_dens = (np.trapz(line(self._x.value), self._x.value) * u.Unit(
                    'kg/(s2 * Angstrom)') * SIGMA * f_value * center.to('Angstrom')).to('1/cm2')

                line_kwargs.update({'v_doppler': v_dop,
                                    'column_density': col_dens})

                line_kwargs.update(self._line_defaults)

                # Add the line to the spectrum object
                line = TauProfile(**line_kwargs)

                spectrum.add_line(model=line)
        else:
            logging.info("No absorption features found using line finder.")

        return getattr(spectrum, self._data_type)
