from astropy.modeling import Fittable1DModel, Parameter, models
from astropy.modeling.core import _ModelMeta
from astropy.modeling.models import Linear1D
import astropy.units as u
from astropy.constants import c
import numpy as np
from scipy import special
import six


class AbsorptionMeta(type):
    """
    Meta class for allowing arbitrary numbers of absorption line features in
    spectral models.
    """
    def __call__(cls, lines=None, continuum=None):
        """
        See the `Absorption1D` initializer for details.
        """
        mod_list = []

        if continuum is not None:
            if issubclass(continuum.__class__, Fittable1DModel):
                mod_list.append(continuum)
            elif isinstance(continuum, six.string_types):
                continuum = getattr(models, continuum, 'Linear1D')
                mod_list.append(continuum)
            else:
                raise AttributeError("Unknown continuum type {}.".format(
                    type(continuum)))
        else:
            continuum = Linear1D(slope=0, intercept=1)
            mod_list.append(continuum)

        if lines is not None:
            if isinstance(lines, list):
                mod_list += lines
            elif issubclass(lines, Fittable1DModel):
                mod_list.append(lines)

        abs_mod = np.sum(mod_list)

        mod = type('Spectrum', (abs_mod.__class__, ), {})

        return mod()


@six.add_metaclass(type('CombinedMeta', (_ModelMeta, AbsorptionMeta), {}))
class Absorption1D(Fittable1DModel):
    """
    One dimensional spectral model that allows the addition of an arbitrary
    number of absorption features.
    """
    def __init__(self, lines=None, continuum=None, *args, **kwargs):
        """
        Custom fittable model representing a spectrum object.

        Parameters
        ----------
        lines : list
            List containing spectral line features as instances of the
            :class:`~spectacle.core.lines.Line` class.

        continuum : :class:`~astropy.modeling.Fittable1DModel` or str
            The `continuum` argument can either be a model instance, or a
            string representing a model type (see `Astropy models list
            <http://docs.astropy.org/en/stable/modeling/index.html#
            module-astropy.modeling.functional_models>`_ for options.

            .. note:: Continuum classes instantiating by string reference will
                      be initialized with default parameters.
        """
        super(Absorption1D, self).__init__(*args, **kwargs)


class Spectrum1D(Fittable1DModel):
    """
    Basic container class for spectral data.
    """

    def __init__(self, *args, **kwargs):
        super(Spectrum1D, self).__init__(*args, **kwargs)

    def add_line(self, *args, **kwargs):
        new_line = AbsorptionLine1D(*args, **kwargs)

        self.__base__ = type('Absorption1D', ())


class AbsorptionLine1D(Fittable1DModel):
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
    lambda_0 = Parameter(fixed=True, min=0)
    f_value = Parameter(fixed=True, min=0, max=2.0)
    gamma = Parameter(fixed=True, min=0)
    v_doppler = Parameter(default=1e5, min=0)
    column_density = Parameter(default=13, min=0, max=25)
    delta_v = Parameter(default=0, fixed=False)
    delta_lambda = Parameter(default=0, fixed=True)

    def evaluate(self, x, lambda_0, f_value, gamma, v_doppler, column_density,
                 delta_v, delta_lambda):
        charge_proton = u.Quantity(4.8032056e-10, 'esu')
        tau_factor = ((np.sqrt(np.pi) * charge_proton ** 2 /
                       (u.M_e.cgs * c.cgs))).cgs

        # Make the input parameters quantities so we can keep track
        # of units
        lambda_bins = x * u.Unit('Angstrom')
        lambda_0 = lambda_0 * u.Unit('Angstrom')
        v_doppler = v_doppler * u.Unit('cm/s')
        column_density = 10 ** column_density * u.Unit('1/cm2')
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
        x = c.c.cgs / v_doppler * (shifted_lambda / lambda_bins - 1.0)
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
