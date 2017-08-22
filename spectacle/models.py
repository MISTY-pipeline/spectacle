from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.models import Voigt1D, Linear1D, Scale
import astropy.units as u
from astropy.constants import c
import numpy as np
from scipy import special
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.modeling.fitting import LevMarLSQFitter
from scipy.integrate import quad
import peakutils


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

    input_units = {'x': u.Angstrom}
    input_units_strict = True

    lambda_0 = Parameter(fixed=True, min=0, unit=u.Angstrom)
    f_value = Parameter(fixed=True, min=0, max=2.0)
    gamma = Parameter(fixed=True, min=0)
    v_doppler = Parameter(default=1e5, min=0, unit=u.Unit('cm/s'))
    column_density = Parameter(default=13, min=0, max=25, unit=u.Unit('1/cm2'))
    delta_v = Parameter(default=0, fixed=False, unit=u.Unit('cm/s'))
    delta_lambda = Parameter(default=0, fixed=True, unit=u.Angstrom)

    def __init__(self, *args, **kwargs):
        super(TauProfile, self).__init__(*args, **kwargs)
        self._fwhm = None
        self._dv90 = None
        self._shifted_lambda = None

    def evaluate(self, x, lambda_0, f_value, gamma, v_doppler, column_density,
                 delta_v, delta_lambda):
        charge_proton = u.Quantity(4.8032056e-10, 'esu')
        tau_factor = ((np.sqrt(np.pi) * charge_proton ** 2 /
                       (u.M_e.cgs * c.cgs))).cgs

        # shift lambda_0 by delta_v
        shifted_lambda = lambda_0 * (1 + delta_v / c.cgs) + delta_lambda
        self._shifted_lambda = shifted_lambda

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

    def fwhm(self, x):
        shifted_lambda = self._shifted_lambda.value if not velocity \
            else VelocityConvert(center=self.lambda_0)(self._shifted_lambda.value)

        mod = ExtendedVoigt1D(x_0=shifted_lambda)
        fitter = LevMarLSQFitter()

        fit_mod = fitter(mod, x, self(x) if not velocity else (
            VelocityConvert(center=self.lambda_0) | self())(x))
        fwhm = fit_mod.fwhm

        return fwhm

    def dv90(self, x):
        tau_tot = quad(self, x[0], x[-1])

        fwhm = self.fwhm(x)

        mn = (np.abs(x - (self.lambda_0 - fwhm))).argmin()
        mx = (np.abs(x - (self.lambda_0 + fwhm))).argmin()

        tau_fwhm = quad(self, x[mn], x[mx])

        while tau_fwhm[0]/tau_tot[0] > 0.9:
            print(tau_fwhm[0]/tau_tot[0])
            mn += 1
            mx -= 1
            tau_fwhm = quad(self, x[mn], x[mx])

        return tau_fwhm, fwhm[0]/tau_tot[0]

    def mask(self, x):
        fwhm = self.fwhm(x)
        mask = (x >= (self.lambda_0 - fwhm)) & (x <= (self.lambda_0 - fwhm))

        return mask


class ExtendedVoigt1D(Voigt1D):
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
        fwhm = 0.5346 * self.fwhm_L + np.sqrt(0.2166 * (self.fwhm_L ** 2.)
                                              + self.fwhm_G ** 2.)

        return fwhm


class SmartScale(Scale):
    @staticmethod
    def evaluate(x, factor):
        """One dimensional Scale model function"""
        if isinstance(factor, u.Quantity):
            return_unit = factor.unit
            factor = factor.value
        if isinstance(x, u.Quantity):
            return (x.value * factor)
        else:
            return factor * x


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


class LSFKernel1D(Fittable1DModel):
    inputs = ('y',)
    outputs = ('y',)

    @staticmethod
    def evaluate(self, y, *args, **kwargs):
        pass


class LineFinder(Fittable1DModel):
    inputs = ('x', 'y')
    outputs = ('y',)

    theshold = Parameter(default=0.5)
    min_distance = Parameter(default=30, min=0)

    @staticmethod
    def evaluate(self, x, y, model, threshold, min_distance):
        indexes = peakutils.indexes(y, thres=threshold, min_dist=min_distance)
        peaks_x = peakutils.interpolate(x, y, ind=indexes)

        for peak in peaks_x:
            model.add_line(lambda_0=peak)

        return model(x)