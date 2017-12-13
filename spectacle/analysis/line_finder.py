import logging

import astropy.units as u
import numpy as np
import peakutils
from astropy.constants import c, m_e
from astropy.modeling import Fittable2DModel

from ..core.spectrum import Spectrum1D
from ..modeling import *

from .initializers import Voigt1DInitializer

# (np.pi * np.exp(2)) / (m_e.cgs * c.cgs) * 0.001
AMP_CONST = 8.854187817e-13 * u.Unit('1/cm')
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
    threshold = Parameter(default=0.1, min=0)
    min_distance = Parameter(default=30, min=0.1)
    rel_tol = Parameter(default=1e-2, min=1e-10, max=1)
    abs_tol = Parameter(default=1e-4, min=1e-10, max=1)
    width = Parameter(default=40, min=2)

    input_units = {'x': u.Unit('km/s')}

    def __init__(self, x, y, ion=None, data_type='optical_depth',
                 *args, **kwargs):
        super(LineFinder, self).__init__(*args, **kwargs)

        if data_type not in ('optical_depth', 'flux', 'flux_decrement'):
            logging.error("No available data type named '%s'. Defaulting to"
                          "'optical_depth'.", data_type)
            self._data_type = 'optical_depth'
        else:
            self._data_type = data_type

        self._result_model = None
        self._regions = None
        self._x = x
        self._y = y

        if ion is not None:
            ion = line_registry.with_name(ion)
            self.center = ion['wave'] * line_registry['wave'].unit

    @property
    def input_units_equivalencies(self):
        """
        Unit equivalencies used by `LineFinder`.
        """
        return {'x': [
            (u.Unit('km/s'), u.Unit('Angstrom'),
             lambda x: WavelengthConvert(
                 self.center)(x * u.Unit('km/s')),
             lambda x: VelocityConvert(self.center)(x * u.Unit('Angstrom')))
        ]}

    def evaluate(self, x, y, center, redshift, threshold, min_distance, rel_tol,
                 abs_tol, width):
        """
        Evaluate `LineFinder` model.
        """
        # Astropy fitters strip modeling of their unit information. However, the
        # first iterate of a fitter includes the quantity arrays passed to the
        # call method. If the input array is a quantity, immediately store the
        # quantity unit as a reference for future iterations.
        if isinstance(x, u.Quantity):
            self.input_units = {'x': x.unit}

        x = u.Quantity(x, self.input_units['x'])
        center = center[0]
        redshift = redshift[0]

        ion_info = line_registry.with_lambda(center)

        logging.info(
            "Relative tolerance: %f, Absolute Tolerance: %f", rel_tol, abs_tol)
        logging.info("Threshold: %f, Minimum Distance: %f",
                     threshold, min_distance)
        indexes = peakutils.indexes(y, thres=threshold, min_dist=min_distance)
        logging.info("Found %i peaks.", len(indexes))

        spectrum = Spectrum1D(center=center, redshift=redshift)

        # Get the interesting regions within the data provided. Don't bother
        # calculating more than once, since y doesn't change.
        if self._regions is None:
            self._regions = find_regions(y, rel_tol=rel_tol, abs_tol=abs_tol)

        for ind in indexes:
            peak = u.Quantity(x[ind], x.unit)
            dv = peak.to('km/s', equivalencies=self.input_units_equivalencies['x']) - center.to(
                'km/s', equivalencies=self.input_units_equivalencies['x'])

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

            # Estimate the doppler b parameter
            # v_dop = np.abs((ion_info['osc_str'] / AMP_CONST * ion ** 2 /
            #                 (fit_line.amplitude_L.value) * c.cgs
            #                 ).decompose().to('cm'))

            print(fit_line.amplitude_L, y[ind])

            fwhm_L = (fit_line.fwhm_L * x.unit).to('Angstrom',
                                                   equivalencies=self.input_units_equivalencies['x'])
            print("Gamma: {:g}".format(fwhm_L.to('cm')))
            # v_dop = (fwhm_L * c.cgs / (fit_line.fwhm * center[0])).decompose().to('cm/s')
            # v_dop = (AMP_CONST * c.cgs * ion_info['osc_str'] * center[0] / (fit_line.amplitude_L)).to('cm/s')
            print(center)
            v_dop = (c.cgs * fwhm_L /
                     (2 * center * fit_line.fwhm)).decompose().to('cm/s')
            print(v_dop)
            print("Velocity: {:g}".format(v_dop))

            # Estimate the column density
            # col_dens_over_tau = (v_dop / (TAU_FACTOR * ion_info['osc_str']) /
            #                      center[0]).decompose().to('1/cm2')
            # col_dens_over_tau = (ion_info['osc_str'] * center[0]**2 * c.cgs /
            #                      (center[0] * v_dop * AMP_CONST)).decompose() * u.Unit('1/cm^2')
            # col_dens = (v_dop * 10 * u.Unit('1/cm') / (TAU_FACTOR * ion_info['osc_str'])).to('1/cm2')
            col_dens = y[ind] / (SIGMA * ion_info['osc_str']
                                 * fit_line(x[ind].value))

            print("Column density: {:g}".format(col_dens))

            # Add the line to the spectrum object
            line = TauProfile(lambda_0=center,
                              delta_v=dv,
                              v_doppler=v_dop,
                              column_density=col_dens)

            spectrum.add_line(model=line)
            # spectrum.append(fit_line)

        # Fit the spectrum mode to the given data
        # fitter = LevMarLSQFitter()

        # fitter(spectrum.optical_depth, x, y, maxiter=1000)
        self._result_model = getattr(spectrum, self._data_type)

        return self._result_model(x)

    def _parameter_units_for_data_units(self, input_units, output_units):
        return OrderedDict()

    @property
    def result_model(self):
        """
        Returns the resultant spectrum model that is constructed by the
        `LineFinder`.
        """
        return self._result_model

    @property
    def regions(self):
        """
        Returns the list of tuples specifying the beginning and end indices of
        identified absorption/emission regions.
        """
        return self._regions

    def fit(self, *args, **kwargs):
        fitter = LevMarLSQFitter(*args, **kwargs)
        fit_finder_model = fitter(self, self._x, self._y, self._y)

        return fit_finder_model.result_model
