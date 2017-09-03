from astropy.modeling import Fittable1DModel, Parameter, Fittable2DModel
from astropy.modeling.models import Voigt1D, Linear1D, Scale
import astropy.units as u
from astropy.constants import c
import numpy as np
from scipy import special
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.modeling.fitting import LevMarLSQFitter
from scipy.integrate import quad
import peakutils
from collections import OrderedDict
from .utils import find_nearest
from .registries import line_registry


class IncompleteLineInformation(Exception): pass


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
    f_value = Parameter(fixed=True, min=0, max=2.0, default=0)
    gamma = Parameter(fixed=True, min=0, default=0)
    v_doppler = Parameter(default=1e5, min=0, unit=u.Unit('cm/s'))
    column_density = Parameter(default=13, min=0, max=25, unit=u.Unit('1/cm2'))
    delta_v = Parameter(default=0, fixed=False, unit=u.Unit('cm/s'))
    delta_lambda = Parameter(default=0, fixed=True, unit=u.Angstrom)

    def __init__(self, name=None, lambda_0=None, *args, **kwargs):
        if name is not None:
            line = line_registry.with_name(name)
            lambda_0 = line['wave'] * u.Angstrom
            name = line['name']
        elif lambda_0 is not None:
            ind = find_nearest(line_registry['wave'], lambda_0)
            line = line_registry[ind]
            name = line['name']
        else:
            raise IncompleteLineInformation(
                "Not enough information to construction absorption line "
                "profile. Please provide at least a name or centroid.")

        kwargs.setdefault('f_value', line['osc_str'])
        kwargs.setdefault('gamma', line['gamma'])

        tied = {'f_value': lambda: line_registry[
            find_nearest(line_registry['wave'], self.lambda_0)]['osc_str'],
                'gamma': lambda: line_registry[
            find_nearest(line_registry['wave'], self.lambda_0)]['gamma']}

        super(TauProfile, self).__init__(name=name, lambda_0=lambda_0,
                                         tied=tied, *args, **kwargs)

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

    def fwhm(self):
        shifted_lambda = self.lambda_0 * (1 + self.delta_v / c.cgs) + self.delta_lambda

        mod = ExtendedVoigt1D(x_0=VelocityConvert(center=self.lambda_0)(shifted_lambda))
        fitter = LevMarLSQFitter()

        x = np.linspace(-10000, 10000, 1000) * u.Unit('km/s')

        fit_mod = fitter(mod, x, (WavelengthConvert(center=self.lambda_0)
                                  | self)(x))
        fwhm = fit_mod.fwhm

        return fwhm

    def dv90(self):
        velocity = np.linspace(-10000, 10000, 1000) * u.Unit('km/s')

        def wave_space(val):
            return WavelengthConvert(center=self.lambda_0)(val)

        def vel_space(val):
            return VelocityConvert(center=self.lambda_0)(val)

        shifted_lambda = self.lambda_0 * (1 + self.delta_v / c.cgs) + self.delta_lambda

        cind = find_nearest(velocity.value, vel_space(shifted_lambda).value)
        mn = mx = cind

        tau_fwhm = quad(lambda x: self(x * u.Angstrom),
                        wave_space(velocity[mn]).value,
                        wave_space(velocity[mx]).value)

        tau_tot = quad(lambda x: self(x * u.Angstrom),
                       wave_space(velocity[0]).value,
                       wave_space(velocity[-1]).value)

        while tau_fwhm[0]/tau_tot[0] < 0.9:
            mn -= 1
            mx += 1
            tau_fwhm = quad(lambda x: self(x * u.Angstrom),
                            wave_space(velocity[mn]).value,
                            wave_space(velocity[mx]).value)

        return (velocity[mn], velocity[mx])

    def mask_range(self):
        fwhm = self.fwhm() * u.Unit('km/s')

        return (-fwhm, fwhm)


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
    input_units_strict = True

    center = Parameter(default=0, fixed=True, unit=u.Angstrom)

    @property
    def input_units(self, *args, **kwargs):
        return {'x': u.Unit('Angstrom')}

    @staticmethod
    def evaluate(x, center):
        # ln_lambda = np.log(x) - np.log(center)
        # vel = (c.cgs * ln_lambda).to('km/s').value

        vel = (c.cgs * ((x - center)/x)).to('km/s')

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
    input_units_strict = True

    center = Parameter(default=0, fixed=True, unit=u.Angstrom)

    @property
    def input_units(self, *args, **kwargs):
        return {'x': u.Unit('km/s')}

    @staticmethod
    def evaluate(x, center):
        wav = center * (1 + x / c.cgs)

        return wav


class DispersionConvert(Fittable1DModel):
    outputs = ('x',)
    input_units_strict = True

    center = Parameter(default=0, fixed=True, unit=u.Angstrom)

    @property
    def input_units(self, *args, **kwargs):
        return {'x': u.Unit('km/s')}

    @property
    def input_units_equivalencies(self):
        return {'x': [
            (u.Unit('km/s'), u.Angstrom,
             lambda x: WavelengthConvert(self.center)(x * u.Unit('km/s')),
             lambda x: VelocityConvert(self.center)(x * u.Angstrom))
        ]}

    def evaluate(self, x, center, *args, **kwargs):
        return x.to('Angstrom', equivalencies=self.input_units_equivalencies['x'])

    def _parameter_units_for_data_units(self, input_units, output_units):
        return OrderedDict([('center', u.Angstrom)])



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


class Masker(Fittable1DModel):
    inputs = ('x',)
    outputs = ('x',)

    def __init__(self, mask_ranges, *args, **kwargs):
        super(Masker, self).__init__(*args, **kwargs)
        self._mask_ranges = mask_ranges

    def evaluate(self, x):
        # [print(rn) for rn in mask_ranges.value]
        masks = [(x >= rn[0]) & (x <= rn[1]) for rn in self._mask_ranges]

        mask = np.logical_or.reduce(masks)

        return x[mask]


class LSFKernel1D(Fittable1DModel):
    inputs = ('y',)
    outputs = ('y',)

    @staticmethod
    def evaluate(self, y, *args, **kwargs):
        pass


class LineFinder(Fittable2DModel):
    inputs = ('x', 'y')
    outputs = ('y',)
    input_units_strict = True

    threshold = Parameter(default=0.5)
    min_distance = Parameter(default=30, min=0)
    width = Parameter(default=10, min=0)
    center = Parameter(default=0, fixed=True, unit=u.Angstrom)

    @property
    def input_units(self, *args, **kwargs):
        return {'x': u.Unit('km/s')}

    @property
    def input_units_equivalencies(self):
        return {'x': [
            (u.Unit('km/s'), u.Angstrom,
             lambda x: WavelengthConvert(self.center)(x * u.Unit('km/s')),
             lambda x: VelocityConvert(self.center)(x * u.Angstrom))
        ]}

    def __init__(self, line_list=None, *args, **kwargs):
        super(LineFinder, self).__init__(*args, **kwargs)
        self._model = None
        self._line_list = line_list

    def evaluate(self, x, y, threshold, min_distance, width, center):
        from .spectrum import SpectrumModel

        model = SpectrumModel(center=center)

        indexes = peakutils.indexes(y, thres=threshold, min_dist=min_distance)
        print(x.value)
        peaks_x = peakutils.interpolate(x.value, y, ind=indexes, width=width)

        for peak in peaks_x:
            peak.to('Angstrom', equivalencies=self.input_units_equivalencies['x'])
            model.add_line(lambda_0=peak, line_list=self._line_list)

        fitter = LevMarLSQFitter()
        self._model = fitter(model.tau, x, y)

        return self._model(x)

    # def _parameter_units_for_data_units(self, input_units, output_units):
    #     return OrderedDict([('min_distance', input_units['x']),
    #                         ('width', input_units['x'])])

    @property
    def model(self):
        return self._model