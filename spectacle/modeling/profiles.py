import logging
from collections import OrderedDict

import astropy.units as u
import numpy as np
from astropy.constants import c, m_e
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Voigt1D
from scipy import special
from scipy.integrate import quad

from ..io.registries import line_registry
from ..utils import find_nearest
from .converters import VelocityConvert, WavelengthConvert

__all__ = ['TauProfile', 'ExtendedVoigt1D']

PROTON_CHARGE = u.Quantity(4.8032056e-10, 'esu')
TAU_FACTOR = ((np.sqrt(np.pi) * PROTON_CHARGE ** 2 /
               (m_e.cgs * c.cgs))).cgs


class IncompleteLineInformation(Exception):
    pass


class LineNotFound(Exception):
    pass


class TauProfile(Fittable1DModel):
    """
    Implements a Voigt profile astropy model. This model generates optical
    depth profiles for absorption features.

    Parameters
    ----------
    lambda_0 : float
       Central wavelength in Angstroms.
    f_value : float
       Absorption line oscillator strength.
    gamma : float
       Absorption line gamma value.
    v_doppler : float
       Doppler b-parameter in cm/s.
    column_density : float
       Column density in cm^-2.
    delta_v : float
       Velocity offset from lambda_0 in cm/s. Default: None (no shift).
    delta_lambda : float
        Wavelength offset in Angstrom. Default: None (no shift).
    lambda_bins : array-like
        Wavelength array for line deposition in Angstroms. If None, one will
        created using n_lambda and dlambda. Default: None.
    n_lambda : int
        Size of lambda bins to create if lambda_bins is None. Default: 12000.
    dlambda : float
        Lambda bin width in Angstroms if lambda_bins is None. Default: 0.01.

    Returns
    -------
    tau_phi : array-like
        An array of optical depth values.
    """
    inputs = ('x',)
    outputs = ('y',)
    input_units_strict = True

    input_units = {'x': u.Unit('Angstrom')}

    lambda_0 = Parameter(fixed=True, min=0, unit=u.Unit('Angstrom'))
    f_value = Parameter(fixed=True, min=0, max=2.0, default=0)
    gamma = Parameter(fixed=True, min=0, default=0)
    v_doppler = Parameter(default=1e5, min=0, unit=u.Unit('cm/s'))
    column_density = Parameter(
        default=1e13, min=0, max=1e25, unit=u.Unit('1/cm2'))
    delta_v = Parameter(default=0, fixed=False, unit=u.Unit('cm/s'))
    delta_lambda = Parameter(default=0, fixed=True, unit=u.Unit('Angstrom'))

    def __init__(self, name=None, lambda_0=None, line_list=None, *args, **kwargs):
        line_mask = np.in1d(line_registry['name'],
                            [line_registry.correct(n) for n in line_list
                             if line_registry.correct(n) is not None]) \
            if line_list is not None else ~np.in1d(line_registry['name'], [])

        line_table = line_registry[line_mask]

        if name is not None:
            line = line_table.with_name(name)

            if line is None:
                raise LineNotFound("No line with name '{}' in current ion "
                                   "table.".format(name))

            lambda_0 = line['wave'] * u.Unit('Angstrom')
            name = line['name']
        elif lambda_0 is not None:
            lambda_0 = u.Quantity(lambda_0, u.Unit('Angstrom'))
            ind = find_nearest(line_table['wave'], lambda_0.value)
            line = line_table[ind]
            name = line['name']
        else:
            raise IncompleteLineInformation(
                "Not enough information to construction absorption line "
                "profile. Please provide at least a name or centroid.")

        kwargs.setdefault('f_value', line['osc_str'])
        kwargs.setdefault('gamma', line['gamma'])
        kwargs.setdefault('tied', {
            'f_value': lambda mod: line_table[find_nearest(line_table['wave'],
                                                           self.lambda_0)]['osc_str'],
            'gamma': lambda mod: line_table[find_nearest(line_table['wave'],
                                                         self.lambda_0)]['gamma']
        })

        super(TauProfile, self).__init__(name=name, lambda_0=lambda_0,
                                         *args, **kwargs)

    def evaluate(self, x, lambda_0, f_value, gamma, v_doppler, column_density,
                 delta_v, delta_lambda):
        # Astropy fitters do not fully support units on model parameters. In
        # such cases, the units are striped while the model is evaluated, and
        # then added back to the parameters once the fitting is complete. This
        # is terrible for modeling that take advantage of parameter units. Thus,
        # units need to be guaranteed.
        x = u.Quantity(x, self.input_units['x'])
        lambda_0 = u.Quantity(lambda_0, 'Angstrom')
        v_doppler = u.Quantity(v_doppler, 'cm/s')
        column_density = u.Quantity(column_density, '1/cm2')
        delta_lambda = u.Quantity(delta_lambda, 'Angstrom')
        delta_v = u.Quantity(delta_v, 'cm/s')

        # shift lambda_0 by delta_v
        shifted_lambda = lambda_0 * (1 + delta_v / c.cgs) + delta_lambda

        # conversions
        nudop = (v_doppler / shifted_lambda).to('Hz')  # doppler width in Hz

        # tau_0
        tau_x = TAU_FACTOR * column_density * f_value / v_doppler
        tau0 = (tau_x * lambda_0).decompose()

        # dimensionless frequency offset in units of doppler freq
        x = c.cgs / v_doppler * (shifted_lambda / x - 1.0)
        a = gamma / (4.0 * np.pi * nudop)  # damping parameter
        phi = self.voigt(a, x)  # line profile
        tau_phi = tau0 * phi  # profile scaled with tau0
        tau_phi = tau_phi.decompose().value

        return tau_phi

    @classmethod
    def voigt(cls, a, u):
        x = np.asarray(u).astype(np.float64)
        y = np.asarray(a).astype(np.float64)

        return special.wofz(x + 1j * y).real

    @staticmethod
    def fit_deriv(x, x_0, b, gamma, f):
        return [0, 0, 0, 0]

    def _parameter_units_for_data_units(self, input_units, output_units):
        return OrderedDict([('lambda_0', u.Unit('Angstrom')),
                            ('f_value', None),
                            ('gamma', None),
                            ('v_doppler', u.Unit('cm/s')),
                            ('column_density', u.Unit('1/cm2')),
                            ('delta_v', u.Unit('cm/s')),
                            ('delta_lambda', u.Unit('Angstrom'))])

    def fwhm(self):
        shifted_lambda = self.lambda_0 * \
            (1 + self.delta_v / c.cgs) + self.delta_lambda

        mod = ExtendedVoigt1D(x_0=VelocityConvert(
            center=self.lambda_0)(shifted_lambda))
        fitter = LevMarLSQFitter()

        x = np.linspace(-10000, 10000, 1000) * u.Unit('km/s')

        fit_mod = fitter(mod, x, (WavelengthConvert(center=self.lambda_0)
                                  | self)(x))
        fwhm = fit_mod.fwhm

        return fwhm

    @u.quantity_input(x=['length', 'speed'])
    def dv90(self, x=None):
        equivalencies = [(u.Unit('km/s'), u.Unit('Angstrom'),
                          lambda x: WavelengthConvert(
                              self.lambda_0)(x * u.Unit('km/s')),
                          lambda x: VelocityConvert(self.lambda_0)(x * u.Unit('Angstrom')))]

        def _calculate(x):
            x = x or np.linspace(-5000, 5000, 10000) * u.Unit('km/s')

            # Profiles require dispersion to be in wavelength space
            x = x.to(u.Unit('Angstrom'), equivalencies=equivalencies)

            shifted_lambda = self.lambda_0 * \
                (1 + self.delta_v / c.cgs) + self.delta_lambda

            y = self(x)
            mask = [y > 1e-5]
            y = y[mask]
            x = x[mask]

            if y.size > 0:
                y95 = np.percentile(y, 95, interpolation='midpoint')
                y5 = np.percentile(y, 5, interpolation='midpoint')

                w95 = x[find_nearest(y, y95)]
                w5 = x[find_nearest(y, y5)]
            else:
                logging.warning("No reasonable amount of optical depth found in "
                                "spectrum, aborting dv90 calculation.")

                return u.Quantity(0)

            return np.abs((c.cgs * (w95 - w5) / shifted_lambda).to('km/s'))

        return _calculate(x)

    @u.quantity_input(x=['length', 'speed'])
    def equivalent_width(self, x=None):
        equivalencies = [(u.Unit('km/s'), u.Unit('Angstrom'),
                        lambda x: WavelengthConvert(self.lambda_0)(x * u.Unit('km/s')),
                        lambda x: VelocityConvert(self.lambda_0)(x * u.Unit('Angstrom')))]

        input_unit = x.unit if x is not None else u.Unit('km/s')

        def _calculate(x):
            x = x or np.linspace(self.lambda_0 - 25 * u.Unit('Angstrom'), 
                                 self.lambda_0 + 25 * u.Unit('Angstrom'), 
                                 1000)

            # The profile only generates optical depth; create flux
            y = np.exp(-self(x.to('Angstrom', equivalencies=equivalencies)))

            # In this case, continuum is defined to be 1
            continuum = 1

            # Average dispersion in the line region.
            avg_dx = np.mean(x[1:] - x[:-1])

            # Calculate equivalent width
            ew = ((1 - y / continuum) * avg_dx).sum()

            return ew

        return _calculate(x)

    def mask_range(self):
        fwhm = self.fwhm() * u.Unit('km/s')

        return (-fwhm, fwhm)


class ExtendedVoigt1D(Voigt1D):
    x_0 = Parameter(default=0)
    amplitude_L = Parameter(default=1)
    fwhm_L = Parameter(default=2 / np.pi, min=0)
    fwhm_G = Parameter(default=np.log(2), min=0)

    @property
    def fwhm(self):
        """
        Calculates an approximation of the FWHM.

        The approximation is accurate to
        about 0.03% (see http://en.wikipedia.org/wiki/Voigt_profile).

        Returns
        -------
        fwhm : float
            The estimate of the FWHM
        """
        fwhm = 0.5346 * self.fwhm_L + np.sqrt(0.2166 * (self.fwhm_L ** 2)
                                              + self.fwhm_G ** 2)

        return fwhm
