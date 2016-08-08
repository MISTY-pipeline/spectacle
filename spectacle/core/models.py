from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.models import Voigt1D as Voigt1DOrig
from astropy import constants as const
from astropy import units as u

import numpy as np
from scipy import special as spc
import matplotlib.pyplot as plt

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
    f_value = Parameter()
    gamma = Parameter()
    v_doppler = Parameter()
    column_density = Parameter()

    def evaluate(self, x, lambda_0, f_value, gamma, v_doppler, column_density):
        lambda_bins = self.meta.get('lambda_bins', None)
        profile = TauProfile(lambda_0=lambda_0, f_value=f_value,
                             gamma=gamma, v_doppler=v_doppler,
                             column_density=column_density,
                             n_lambda=x.size, lambda_bins=lambda_bins)

        flux = np.exp(-profile.optical_depth)

        return flux

    @staticmethod
    def fit_deriv(x, x_0, b, gamma, f):
        return [0, 0, 0, 0]


if __name__ == '__main__':
    x = np.arange(1000., 1500., 1)
    # v = Voigt1D(x_0=1215.6, b=87.7, gamma=2e-9, f=0.4164)
    for f in np.arange(0.1, 0.8, 0.1):
        v2 = Voigt1D(x_0=1250.0, b=7.7, gamma=2e-7, f=f)
    # vo = Voigt1DOrig(x_0=50, amplitude_L=10, fwhm_L=0.5, fwhm_G=0.25)

    # plt.plot(x, v(x))
        plt.plot(x, v2(x))
    # plt.plot(x, vo(x) * 1e-20)
    plt.show()
