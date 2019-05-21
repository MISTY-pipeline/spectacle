import astropy.units as u
import numpy as np
from astropy.constants import c, m_e
from astropy.modeling import Fittable1DModel, Parameter
from scipy import special

from ..registries.lines import line_registry
from ..utils.misc import find_nearest

__all__ = ['OpticalDepth1D']

PROTON_CHARGE = u.Quantity(4.8032056e-10, 'esu')
TAU_FACTOR = (np.sqrt(np.pi) * PROTON_CHARGE ** 2 / (m_e * c)).cgs


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
    delta_v = Parameter(default=0, min=-500, max=500, fixed=False, unit=u.Unit('km/s'))
    delta_lambda = Parameter(default=0, min=-100, max=100, fixed=False, unit=u.Unit('Angstrom'))

    def __init__(self, name=None, line_list=None, *args, **kwargs):
        super(OpticalDepth1D, self).__init__(*args, **kwargs)

        line_mask = np.in1d(line_registry['name'],
                            [line_registry.correct(n) for n in line_list
                             if line_registry.correct(n) is not None]) \
            if line_list is not None else ~np.in1d(line_registry['name'], [])

        line_table = line_registry[line_mask]

        if name is not None or isinstance(name, str):
            line = line_table.with_name(name)

            if line is None:
                raise LineNotFound("No line with name '{}' in current ion "
                                   "table.".format(name))
        else:
            ind = find_nearest(line_table['wave'].value, self.lambda_0.value)
            line = line_table[ind]

        self.lambda_0 = line['wave']
        self.name = str(line['name'])
        self.f_value = line['osc_str']
        self.gamma = line['gamma']

    def evaluate(self, x, lambda_0, f_value, gamma, v_doppler, column_density,
                 delta_v, delta_lambda):
        with u.set_enabled_equivalencies(u.spectral() + u.doppler_relativistic(lambda_0)):
            x = u.Quantity(x, 'Angstrom')
            vel = u.Quantity(x, 'km/s')

        # Convert the log column density value back to unit-ful value
        column_density = u.Quantity(10 ** column_density, '1/cm2')

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

        phi = OpticalDepth1D.voigt(a.decompose().value,
                                   x.decompose().value)  # line profile
        tau_phi = tau0 * phi  # profile scaled with tau0

        return tau_phi.value

    @staticmethod
    def voigt(a, u):
        return special.wofz(u + 1j * a).real

    @staticmethod
    def fit_deriv(x, x_0, b, gamma, f):
        return [0, 0, 0, 0]
