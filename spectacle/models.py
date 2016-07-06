from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.models import Voigt1D as Voigt1DOrig
from astropy import constants as const
from astropy import units as u

import numpy as np
from scipy import special as spc
import matplotlib.pyplot as plt


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
        L=al/(\pi ((x-mu)^2+gamma^2)) + lin \ x + off

      The Voigt profile is calculated via the real part of the Faddeeva
      function. For details, see http://en.wikipedia.org/wiki/Voigt_profile
      and http://en.wikipedia.org/wiki/Error_function.
    """
    x_0 = Parameter()
    # amp = Parameter()
    b = Parameter()
    gamma = Parameter()
    # sigma = Parameter(default=1.0 / np.sqrt(2.0))
    f = Parameter()

    @staticmethod
    def evaluate(x, x_0, b, gamma, f):
        sigma = 1.0 / np.sqrt(2.0)
        # Set to zero, because evaluation is done in velocity space
        mu = 0.0

        b = b * u.Unit('km/s')
        gamma = gamma * u.Unit('cm')
        x_0 = x_0 * u.Unit('Angstrom')
        x = x * u.Unit('Angstrom')

        # Doppler width
        bl = x_0 * b / const.c
        # The constant equals (pi e^2)/(m_e c^2)
        # I have no idea what units the amplitude is expected to be in
        amp = -8.85282064473e-13 * u.Unit('1/J') * f * x_0 ** 2 / bl
        # A factor of 2.0 because `al` defines the half FWHM in Voigt profile
        al = gamma / bl / 2.0

        x = (x - x_0) / bl
        z = (x - mu) + ((1.j) * abs(al)) / (np.sqrt(2.0) * abs(sigma))
        y = amp * np.real(spc.wofz(z.value))
        y /= (abs(sigma) * np.sqrt(2.0 * np.pi))

        return y.value

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
