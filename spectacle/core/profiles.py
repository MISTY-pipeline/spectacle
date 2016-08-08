import astropy.units as u
from astropy import constants as c
import numpy as np
from scipy.special import wofz


class TauProfile:
    """
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
    def __init__(self, lambda_0, f_value, gamma, v_doppler, column_density,
                 delta_v=None, delta_lambda=None, lambda_bins=None,
                 n_lambda=12000, dlambda=0.01):
        charge_proton = u.Quantity(4.8032056e-10, 'esu')
        tau_factor = ((np.sqrt(np.pi) * charge_proton ** 2 /
                       (u.M_e.cgs * c.c.cgs))).cgs.value

        # shift lambda_0 by delta_v
        if delta_v is not None:
            lam1 = lambda_0 * (1 + delta_v / c.c.cgs.value)
        elif delta_lambda is not None:
            lam1 = lambda_0 + delta_lambda
        else:
            lam1 = lambda_0

        # conversions
        nudop = 1e8 * v_doppler / lam1  # doppler width in Hz

        # create wavelength
        if lambda_bins is None:
            lambda_bins = lam1 + \
                          np.arange(n_lambda, dtype=np.float) * dlambda - \
                          n_lambda * dlambda / 2  # wavelength vector (angstroms)

        # tau_0
        tau_X = tau_factor * column_density * f_value / v_doppler
        tau0 = tau_X * lambda_0 * 1e-8

        # dimensionless frequency offset in units of doppler freq
        x = c.c.cgs.value / v_doppler * (lam1 / lambda_bins - 1.0)
        a = gamma / (4.0 * np.pi * nudop)  # damping parameter
        phi = self.voigt(a, x)  # line profile
        tau_phi = tau0 * phi  # profile scaled with tau0

        self.lambda_bins = lambda_bins
        self.tau_phi = tau_phi

    @property
    def optical_depth(self):
        return self.tau_phi

    @property
    def dispersion(self):
        return self.lambda_bins

    @classmethod
    def voigt(cls, a, u):
        x = np.asarray(u).astype(np.float64)
        y = np.asarray(a).astype(np.float64)

        return wofz(x + 1j * y).real


if __name__ == '__main__':
    tau_profile = TauProfile(977.02, 0.4164, 1e6, 87.3, 5.25e13,
                             delta_lambda=10)
    tau_profile2 = TauProfile(877.02, 0.4164, 10, 1.27e6, 5.25e13)
    # tau_profile3 = TauProfile(977.02, 0.4164, 1e-7, 1.27e6, 5.25e13)

    import matplotlib.pyplot as plt

    f, (ax1, ax2) = plt.subplots(2, 1)

    ax1.step(tau_profile.dispersion, tau_profile.optical_depth)
    plt.step(tau_profile2.lambda_bins, tau_profile2.tau_phi)
    # plt.step(tau_profile3.lambda_bins, tau_profile3.tau_phi)

    mask = (tau_profile.dispersion > 950) & (tau_profile.lambda_bins < 1000)

    ax2.axhline(1.0, linestyle='--')
    ax2.step(tau_profile.dispersion[mask],
             np.exp(-tau_profile.optical_depth[mask]))
    ax2.set_ylim(0.0, 1.0)
    # plt.step(tau_profile2.lambda_bins, np.exp(-tau_profile2.tau_phi))
    # plt.step(tau_profile3.lambda_bins, np.exp(-tau_profile3.tau_phi))
    plt.show()