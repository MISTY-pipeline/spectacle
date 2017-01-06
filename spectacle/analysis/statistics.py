import numpy as np
import uncertainties.unumpy as unp


def fwhm(self, x_0):
    """
    Calculates an approximation of the FWHM.

    The approximation is accurate to
    about 0.03% (see http://en.wikipedia.org/wiki/Voigt_profile).

    Returns
    -------
    FWHM : float
        The estimate of the FWHM
    """
    v_prof = self.get_profile(x_0)

    # The width of the Lorentz profile
    fl = 2.0 * v_prof.gamma

    # Width of the Gaussian [2.35 = 2*sigma*sqrt(2*ln(2))]
    fd = 2.35482 * 1/np.sqrt(2.)

    return 0.5346 * fl + np.sqrt(0.2166 * (fl ** 2.) + fd ** 2.)


def optical_depth(self, x_0):
    """
    Return the optical depth at some wavelength.

    Parameters
    ----------
    x_0 : float
        Line center from which to calculate tau.

    Returns
    -------
    tau : float
        The value of the optical depth at the given wavelength.
    """
    flux = unp.uarray(self.data, self.uncertainty)
    idx = (np.abs(self.dispersion - x_0)).argmin()
    tau = unp.log(1.0/flux[idx])

    return unp.nominal_values(tau), unp.std_devs(tau)


def centroid(self, x_0):
    """
    Return the centroid for Voigt profile near the given wavelength.

    Parameters
    ----------
    x_0 : float
        Wavelength new the given profile from which to calculate the
        centroid.

    Returns
    -------
    cent : float
        The centroid of the profile.
    """
    disp = self.dispersion
    flux = unp.uarray(self.data, self.uncertainty)

    cent = np.trapz(disp * flux, disp) / np.trapz(flux, disp)

    return unp.nominal_values(cent), unp.std_devs(cent)


def equivalent_width(self, x_range=None, x_0=None, line_name=None):
    if x_range is not None and (isinstance(x_range, list) or
                                    isinstance(x_range, tuple)):
        x1, x2 = x_range
    elif x_0 is not None or line_name is not None:
        region_mask = self._get_range_mask(x_0)
        region_disp = self.dispersion[region_mask]
        x1, x2 = region_disp[0], region_disp[-1]
    else:
        x1, x2 = self.dispersion[0], self.dispersion[-1]

    mask = (self.dispersion >= x1) & (self.dispersion <= x2)
    disp = self.dispersion[mask]
    flux = self.data[mask]
    uncert = self.uncertainty[mask]

    # Compose the uncertainty array
    uflux = unp.uarray(flux, uncert)

    # Continuum is always assumed to be 1.0
    avg_cont = 1.0

    # Average dispersion in the line region.
    avg_dx = np.mean(disp[1:] - disp[:-1])

    # Calculate equivalent width
    ew = ((avg_cont - uflux) * (avg_dx / avg_cont)).sum()

    return ew.nominal_value, ew.std_dev