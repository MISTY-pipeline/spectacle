import astropy.units as u
import numpy as np
from astropy.constants import c
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.models import (RedshiftScaleFactor, Scale)

from scipy import special


class InjectionMeta(type):
    def __call__(cls, spectrum, *args, **kwargs):
        return type('OpticalDepth', (RedshiftScaleFactor |
                                     spectrum.absorber_model | Scale,),
                    dict(OpticalDepth.__dict__))


class OpticalDepth(metaclass=InjectionMeta):
    inputs = ('x',)
    outputs = ('y',)

    input_units_strict = True

    input_units = {'x': u.Unit('Angstrom')}


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

    def evaluate(self, x, lambda_0, f_value, gamma, v_doppler, column_density,
                 delta_v, delta_lambda):
        charge_proton = u.Quantity(4.8032056e-10, 'esu')
        tau_factor = ((np.sqrt(np.pi) * charge_proton ** 2 /
                       (u.M_e.cgs * c.cgs))).cgs

        # shift lambda_0 by delta_v
        shifted_lambda = lambda_0 * (1 + delta_v / c.cgs) + delta_lambda

        # conversions
        nudop = (v_doppler / shifted_lambda).to('Hz')  # doppler width in Hz

        # tau_0
        tau_x = tau_factor * column_density * f_value / v_doppler
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
