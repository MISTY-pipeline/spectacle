from astropy.modeling import Fittable1DModel, Parameter
import numpy as np

from .profiles import TauProfile


class Voigt1D(Fittable1DModel):
    """
      Implements a Voigt profile (convolution of Cauchy-Lorentz
      and Gaussian distribution).

      .. note:: The profile is implemented so that `al` is half
                the FWHM.

      *Fit parameters*:
        - `amp` - Amplitude
        - `al` - Scale parameter of the Cauchy-Lorentz distribution
        - `sigma` - The width of the Gaussian (usually called sigma)
        - `mu` - Center
        - `off` - Constant offset
        - `lin` - Linear contribution

      x_0      Wavelength of the transition                     A
      b        Doppler parameter (corresponds                   km/s
               to sqrt(2) times the velocity dispersion).
      gamma    Damping width (full width at half maximum of
               the Lorentzian)                                  cm
      f        Oscillator strength (unitless)                   --

      Notes
      -----
      The Voigt profile V is defined as the convolution

      .. math::
        V = A\int G(x')L(x-x')dx'

      of a Gaussian distribution

      .. math::
        G=1/(2 \pi \ sigma)\exp(-(x-mu)^2/(2 \ sigma^2))

      and a Cauchy-Lorentz distribution

      .. math::
        L=al/(\pi ((x-mu)^2+gamma^2))

      The Voigt profile is calculated via the real part of the Faddeeva
      function. For details, see http://en.wikipedia.org/wiki/Voigt_profile
      and http://en.wikipedia.org/wiki/Error_function.
    """
    lambda_0 = Parameter()
    f_value = Parameter(min=0.0, max=1.0)
    gamma = Parameter()
    v_doppler = Parameter()
    column_density = Parameter(min=1e10, max=1e30)

    def evaluate(self, x, lambda_0, f_value, gamma, v_doppler, column_density):
        lambda_bins = self.meta.get('lambda_bins', None)
        profile = TauProfile(lambda_0=lambda_0, f_value=f_value,
                             gamma=gamma, v_doppler=v_doppler,
                             column_density=column_density,
                             n_lambda=x.size, lambda_bins=lambda_bins)

        flux = np.exp(-profile.optical_depth) - 1.0

        return flux

    @staticmethod
    def fit_deriv(x, x_0, b, gamma, f):
        return [0, 0, 0, 0]
