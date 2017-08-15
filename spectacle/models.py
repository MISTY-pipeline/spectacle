from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.core import _ModelMeta
from astropy.modeling.models import Linear1D, RedshiftScaleFactor
import astropy.units as u
from astropy.constants import c
import numpy as np
from scipy import special
import six


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

    lambda_0 = Parameter(fixed=True, min=0)
    f_value = Parameter(fixed=True, min=0, max=2.0)
    gamma = Parameter(fixed=True, min=0)
    v_doppler = Parameter(default=1e5, min=0)
    column_density = Parameter(default=13, min=0, max=25)
    delta_v = Parameter(default=0, fixed=False)
    delta_lambda = Parameter(default=0, fixed=True)

    @classmethod
    def evaluate(cls, x, lambda_0, f_value, gamma, v_doppler, column_density,
                 delta_v, delta_lambda):
        charge_proton = u.Quantity(4.8032056e-10, 'esu')
        tau_factor = ((np.sqrt(np.pi) * charge_proton ** 2 /
                       (u.M_e.cgs * c.cgs))).cgs

        # Make the input parameters quantities so we can keep track
        # of units
        lambda_bins = x * u.Unit('Angstrom')
        lambda_0 = lambda_0 * u.Unit('Angstrom')
        v_doppler = v_doppler * u.Unit('cm/s')
        column_density = column_density * u.Unit('1/cm2')
        delta_v = (delta_v or 0) * u.Unit('cm/s')
        delta_lambda = (delta_lambda or 0) * u.Unit('Angstrom')

        # shift lambda_0 by delta_v
        shifted_lambda = lambda_0 * (1 + delta_v / c.cgs) + delta_lambda

        # conversions
        nudop = (v_doppler / shifted_lambda).to('Hz')  # doppler width in Hz

        # tau_0
        tau_x = tau_factor * column_density * f_value / v_doppler
        tau0 = (tau_x * lambda_0).decompose()

        # dimensionless frequency offset in units of doppler freq
        x = c.cgs / v_doppler * (shifted_lambda / lambda_bins - 1.0)
        a = gamma / (4.0 * np.pi * nudop)  # damping parameter
        phi = cls.voigt(a, x)  # line profile
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


class VelocityConvert(Fittable1DModel):
    """
    Model to convert from wavelength space to velocity space.

    Parameters
    ----------
    center : float
        Central wavelength.

    Notes
    -----
    Model formula:

        .. math:: v = \frac{\lambda - \lambda_c}{\lambda} c
    """
    inputs = ('x',)
    outputs = ('x',)

    center = Parameter(default=0, fixed=True)

    @staticmethod
    def evaluate(x, center):
        # ln_lambda = np.log(x) - np.log(center)
        # vel = (c.cgs * ln_lambda).to('km/s').value

        vel = (c.cgs * ((x - center)/x)).to('km/s').value

        return vel


class WavelengthConvert(Fittable1DModel):
    """
    Model to convert from velocity space to wavelength space.

    Parameters
    ----------
    center : float
        Central wavelength.

    Notes
    -----
    Model formula:

        .. math:: \lambda = \lambda_c (1 + \frac{\lambda}{c}
    """
    inputs = ('x',)
    outputs = ('x',)

    center = Parameter(default=0, fixed=True)

    @staticmethod
    def evaluate(x, center):
        vel = x * u.Unit('km/s')
        wav = center * (1 + vel / c.cgs).value

        return wav


class FluxConvert(Fittable1DModel):
    inputs = ('y',)
    outputs = ('y',)

    @staticmethod
    def evaluate(y):
        return np.exp(-y) - 1


class FluxDecrementConvert(Fittable1DModel):
    inputs = ('y',)
    outputs = ('y',)

    @staticmethod
    def evaluate(y):
        return 1- np.exp(-y) - 1