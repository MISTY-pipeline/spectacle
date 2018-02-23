import logging
from collections import OrderedDict

import astropy.units as u
import numpy as np
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
from ..utils import peak

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
    threshold = Parameter(default=0.1, min=0)
    min_distance = Parameter(default=2, min=0.1)
    width = Parameter(default=15, min=2)

    input_units = {'x': u.Unit('km/s')}

    def __init__(self, x, y, ion_name=None, data_type='optical_depth',
                 defaults=None, *args, **kwargs):
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
        # self.input_units['x'] = x.unit
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

    def evaluate(self, x, y, center, redshift, threshold, min_distance, width):
        """
        Evaluate `LineFinder` model.
        """
        # Units are stripped in the evaluate methods of models
        x = u.Quantity(x, unit=self.input_units['x'])

        # Convert the min_distance from dispersion units to data elements
        min_ind = (np.abs(x.value - (x[0].value + min_distance))).argmin()

        logging.info("Min distance: %i elements.", min_ind)

        # Take a first iteration of the minima finder
        indicies = peak.indexes(np.max(y) - y, thres=threshold, min_dist=min_ind)

        # Given each minima in the finder, construct a new spectrum object with
        # absorption lines at the given indicies.
        spectrum = Spectrum1D(center=self.center, redshift=self.redshift)

        for ind in indicies:
            redshifted_center = u.Quantity(x[ind], x.unit)
            deredshifted_peak = spectrum._redshift_model.inverse(redshifted_center)
            
            # Get the index of this line model's centroid
            ind = (np.abs(x.value - redshifted_center.value)).argmin()

            # Construct a dictionary with all of the true values for this
            # absorption line.
            line_kwargs = dict(lambda_0=center)

            # Based on the dispersion type, calculate the lambda or velocity
            # deltas.
            if x.unit.physical_type == 'length':
                    line_kwargs['delta_lambda'] = deredshifted_peak.to('Angstrom') - center.to(
                        'Angstrom')
            elif x.unit.physical_type == 'speed':
                line_kwargs['delta_v'] = deredshifted_peak.to(
                    'km/s') - center.to('km/s')
            else:
                logging.error("Could not get physical type of dispersion "
                                "axis unit.")

            # Calculate some initial parameters for velocity and col density
            line_kwargs.update(estimate_line_parameters(x, y, ind, min_ind))

            # Create a line profile model and add it to the spectrum object
            spectrum.add_line(**line_kwargs)

        logging.info("Found %i minima.", len(indicies))

        return getattr(spectrum, self._data_type)(x)

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

    def find_regions(self):
        """
        Returns the list of tuples specifying the beginning and end indices of
        identified absorption/emission regions.
        """
        return find_regions(self._y, rel_tol=1e-2, abs_tol=1e-4)

    def fit(self, *args, **kwargs):
        fitter = LevMarLSQFitter()  # MCMCFitter() #
        fit_finder = fitter(self, self._x, self._y, self._y)

        return fit_finder.result_model


def estimate_line_parameters(x, y, ind, min_ind):
    bound_low = max(0, min(ind - min_ind, x.size - 1))
    bound_up = max(0, min(ind + min_ind, x.size - 1))
    mask = ((x > x[bound_low]) & (x < x[bound_up]))

    x = x[mask]
    y = y[mask]

    # width can be estimated by the weighted 2nd moment of the x coordinate.
    dx = x - np.mean(x)
    fwhm = 2 * np.sqrt(np.sum((dx * dx) * y) / np.sum(y))

    # amplitude is derived from area.
    delta_x = x[1:] - x[:-1]
    sum_y = np.sum((y[1:]) * delta_x)
    height = sum_y / (fwhm / 2.355 * np.sqrt( 2 * np.pi))

    # Estimate the doppler b parameter
    v_dop = 0.60056120439322491 * fwhm

    # Estimate the column density
    col_dens = sum_y.value * u.Unit('1/cm2')

    logging.info("\tb={}\n\tN={}".format(v_dop, col_dens))

    return dict(v_doppler=v_dop, column_density=col_dens)