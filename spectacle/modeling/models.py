from astropy.modeling import Fittable1DModel, Parameter, models
from astropy.modeling.core import _ModelMeta

from astropy.modeling.models import Linear1D
import numpy as np

import astropy.units as u
from astropy import constants as c
from scipy import special

import six


class Voigt1D(Fittable1DModel):
    """
    Implements a Voigt profile (convolution of Cauchy-Lorentz and Gaussian
    distribution).

    Create an optical depth vs. wavelength profile for an
    absorption line using a voigt profile. This follows the paradigm of
    the :func:`~yt.analysis_modules.absorption_spectrum.absorption_line`
    profile generator.

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
        Wavelength array for line deposition in Angstroms. If None, one will be
        created using n_lambda and dlambda. Default: None.
    n_lambda : int
        Size of lambda bins to create if lambda_bins is None. Default: 12000.
    dlambda : float
        Lambda bin width in Angstroms if lambda_bins is None. Default: 0.01.
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
                       (u.M_e.cgs * c.c.cgs))).cgs

        # Make the input parameters quantities so we can keep track
        # of units
        lambda_bins = x * u.Unit('Angstrom')
        lambda_0 = lambda_0 * u.Unit('Angstrom')
        v_doppler = v_doppler * u.Unit('cm/s')
        column_density = 10 ** column_density * u.Unit('1/cm2')
        delta_v = (delta_v or 0) * u.Unit('cm/s')
        delta_lambda = (delta_lambda or 0) * u.Unit('Angstrom')

        # shift lambda_0 by delta_v
        shifted_lambda = lambda_0 * (1 + delta_v / c.c.cgs) + delta_lambda

        # conversions
        nudop = (v_doppler / shifted_lambda).to('Hz')  # doppler width in Hz

        # tau_0
        tau_X = tau_factor * column_density * f_value / v_doppler
        tau0 = (tau_X * lambda_0).decompose()

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
            continuum = Linear1D(slope=0, intercept=1,
                                 # fixed={'slope': True, 'intercept': True},
                                 bounds={'slope': (-100, 100),
                                         'intercept': (0, 2)}
                                 )
            mod_list.append(continuum)

        if lines is not None:
            if isinstance(lines, list):
                mod_list += lines
            elif issubclass(lines, Fittable1DModel):
                mod_list.append(lines)

        abs_mod = np.sum(mod_list)

        def call(self, dispersion, uncertainty=None, dispersion_unit=None,
                 *args, **kwargs):
            from ..core.spectra import Spectrum1D

            if hasattr(abs_mod, '_submodels'):
                mods = [x for x in abs_mod]
                cont, lines = mods[0], mods[1:]
            else:
                cont, lines = abs_mod, []

            flux = super(abs_mod.__class__, self).__call__(dispersion,
                                                           *args, **kwargs)

            spectrum = Spectrum1D(flux, dispersion=dispersion,
                                  dispersion_unit=dispersion_unit, lines=lines,
                                  continuum=cont(dispersion))

            return spectrum

        mod = type('Absorption1D', (abs_mod.__class__, ), {'__call__': call})

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

    def get_range_mask(self, dispersion, name=None):
        """
        Returns a mask for the model spectrum that indicates where the values
        are discernible from the continuum. I.e. where the absorption data
        is contained.

        If a `x_0` value is provided, only the related Voigt profile will be
        considered, otherwise, the entire model is considered.

        Parameters
        ----------
        dispersion : array-like
            The x values for which the model spectrum will be calculated.
        name : str, optional
            The line name used to grab a specific Voigt profile.

        Returns
        -------
        array-like
            Boolean array indicating indices of interest.
        """
        profile = self.model if name is None else self.get_profile(name)
        vdisp = profile(dispersion)
        cont = np.zeros(dispersion.shape)

        return ~np.isclose(vdisp, cont, rtol=1e-2, atol=1e-5)

    def get_profile(self, name):
        """
        Retrieve the particular :class:`spectacle.modeling.models.Voigt1D`
        model for a given absorption feature.

        Parameters
        ----------
        name : str
            The unique name of the model.

            .. note:: Name uniqueness is not enforced. If there is more than
                      one line with the same name, only the first will be
                      returned.

        Returns
        -------
        :class:`spectacle.modeling.models.Voigt1D`
            The Voigt model for the particular line.
        """
        return next((sm for sm in self._submodels if sm.name == name), None)

    def get_fwhm(self, line_name):
        """
        Return the full width at half max for a given absorption line feature.

        Parameters
        ----------
        line_name : str
            The name of the absorption line feature.

        Returns
        -------
        float
            The calculated full width at half max.
        """
        profile = self.get_profile(line_name)

        return profile.fwhm
