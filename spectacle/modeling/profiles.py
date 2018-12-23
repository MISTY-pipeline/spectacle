import logging
from collections import OrderedDict

import astropy.units as u
import numpy as np
from astropy.constants import c, m_e
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Voigt1D, Gaussian1D
from scipy import special

from ..registries.lines import line_registry
from ..utils.misc import find_nearest, DOPPLER_CONVERT

__all__ = ['OpticalDepth1D']

PROTON_CHARGE = u.Quantity(4.8032056e-10, 'esu')
TAU_FACTOR = ((np.sqrt(np.pi) * PROTON_CHARGE ** 2 /
               (m_e.cgs * c.cgs))).cgs


class IncompleteLineInformation(Exception):
    pass


class LineNotFound(Exception):
    pass


class OpticalDepth1D(Fittable1DModel):
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
       Doppler b-parameter in km/s.
    column_density : float
       Column density in cm^-2.
    delta_v : float
       Velocity offset from lambda_0 in km/s. Default: None (no shift).
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

    lambda_0 = Parameter(fixed=True, min=0, default=1215.6701, unit=u.Unit('Angstrom'))
    f_value = Parameter(fixed=True, min=0, max=2.0, default=0)
    gamma = Parameter(fixed=True, min=0, default=0)
    v_doppler = Parameter(default=10, min=.1, max=1e5, unit=u.Unit('km/s'))
    column_density = Parameter(default=13, min=8, max=25)
    delta_v = Parameter(default=0, min=0, fixed=False, unit=u.Unit('km/s'))
    delta_lambda = Parameter(default=0, min=-100, max=100, fixed=True, unit=u.Unit('Angstrom'))

    def __init__(self, name=None, line_list=None, *args, **kwargs):
        super(OpticalDepth1D, self).__init__(*args, **kwargs)

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

            name = line['name']
            self.lambda_0 = line['wave']
        else:
            ind = find_nearest(line_table['wave'].value, self.lambda_0.value)
            line = line_table[ind]
            name = line['name']

        self.name = name
        self.f_value = line['osc_str']
        self.gamma = line['gamma']
        self._velocity_convention = 'relativistic'

    @property
    def velocity_convention(self):
        return self._velocity_convention

    @velocity_convention.setter
    def velocity_convention(self, value):
        if value in DOPPLER_CONVERT.keys():
            self._velocity_convention = value
        else:
            raise ValueError("Velocity convention must be one of {}.".format(
                DOPPLER_CONVERT.keys()))

    def evaluate(self, x, lambda_0, f_value, gamma, v_doppler, column_density,
                 delta_v, delta_lambda):
        lambda_0 = u.Quantity(lambda_0, 'Angstrom')
        v_doppler = u.Quantity(v_doppler, 'km/s')
        column_density = u.Quantity(10 ** column_density, '1/cm2')
        delta_lambda = u.Quantity(delta_lambda, 'Angstrom')
        delta_v = u.Quantity(delta_v, 'km/s')

        with u.set_enabled_equivalencies(DOPPLER_CONVERT[self.velocity_convention](lambda_0)):
            x = u.Quantity(x, 'Angstrom')

        # shift lambda_0 by delta_v
        shifted_lambda = lambda_0 * (1 + delta_v / c.cgs) + delta_lambda

        # conversions
        nudop = (v_doppler / shifted_lambda).to('Hz')  # doppler width in Hz

        # tau_0
        tau_x = TAU_FACTOR * column_density * f_value / v_doppler
        tau0 = (tau_x * lambda_0).decompose()[0]

        # dimensionless frequency offset in units of doppler freq
        x = c.cgs / v_doppler.to('cm/s') * (shifted_lambda / x - 1.0)
        a = gamma / (4.0 * np.pi * nudop)  # damping parameter
        phi = OpticalDepth1D.voigt(a, x)  # line profile
        tau_phi = tau0 * phi  # profile scaled with tau0
        tau_phi = tau_phi.decompose().value

        return tau_phi

    @staticmethod
    def voigt(a, u):
        x = np.asarray(u).astype(np.float64)
        y = np.asarray(a).astype(np.float64)

        return special.wofz(x + 1j * y).real

    @staticmethod
    def fit_deriv(x, x_0, b, gamma, f):
        return [0, 0, 0, 0]

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return OrderedDict([('lambda_0', u.Unit('Angstrom')),
                            ('f_value', None),
                            ('gamma', None),
                            ('v_doppler', u.Unit('km/s')),
                            ('column_density', None),
                            ('delta_v', u.Unit('km/s')),
                            ('delta_lambda', u.Unit('Angstrom'))])

    def fwhm(self, x):
        """
        It's unclear how to get the fwhm given the parameters of this Voigt
        model, so instead, fit a Gaussian to this model and calculate the fwhm
        from that.
        """
        y = self(x)  # Generate optical depth values from this model

        dx = x - np.mean(x)
        fwhm = 2 * np.sqrt(np.sum((dx * dx) * y) / np.sum(y))
        center = np.sum(x * y) / np.sum(y)
        sigma = fwhm / 2.355

        # Amplitude is derived from area
        delta_x = x[1:] - x[:-1]
        sum_y = np.sum(y[1:] * delta_x)

        height = sum_y / (sigma * np.sqrt(2 * np.pi))

        g = Gaussian1D(amplitude=height,
                       mean=center,
                       stddev=sigma,
                       bounds={'mean': (x[0].value, x[-1].value),
                               'stddev': (None, 4 * sigma.value)})

        g_fit = LevMarLSQFitter()(g, x, y)

        return g_fit.fwhm

    # @u.quantity_input(x=['length', 'speed'])
    # def delta_v_90(self, x):
    #     return delta_v_90(x=x, y=self(x),
    #                       rest_wavelength=self.lambda_0.quantity)
    #
    # @u.quantity_input(x=['length', 'speed'])
    # def equivalent_width(self, x):
    #     return equivalent_width(x=x, y=self(x))


# class ExtendedVoigt1D(Voigt1D):
#     x_0 = Parameter(default=0)
#     amplitude_L = Parameter(default=1)
#     fwhm_L = Parameter(default=2 / np.pi, min=0)
#     fwhm_G = Parameter(default=np.log(2), min=0)
#
#     @property
#     def fwhm(self):
#         """
#         Calculates an approximation of the FWHM.
#
#         The approximation is accurate to
#         about 0.03% (see http://en.wikipedia.org/wiki/Voigt_profile).
#
#         Returns
#         -------
#         fwhm : float
#             The estimate of the FWHM
#         """
#         fwhm = 0.5346 * self.fwhm_L + np.sqrt(0.2166 * (self.fwhm_L ** 2) +
#                                               self.fwhm_G ** 2)
#
#         return fwhm
