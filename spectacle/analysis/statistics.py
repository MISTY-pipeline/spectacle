import astropy.units as u
import numpy as np
import logging
from ..utils.misc import find_nearest
from specutils.analysis.width import _compute_single_fwhm
from scipy.interpolate import UnivariateSpline

__all__ = ['delta_v_90', 'full_width_half_max', 'equivalent_width']


@u.quantity_input(x=['length', 'speed'])
def delta_v_90(x, y, continuum=None):
    """
    Calculate the dispersion that encompasses the central 90 percent of the
    apparant optical depth. Follows the formulation defined in Prochaska &
    Wolf (1997).

    Parameters
    ----------
    x : :class:`~astropy.units.Quantity`
        The dispersion axis. Can be either wavelength or velocity space.
    y : array-like
        Flux or optical depth array. Note that the calculation assumes that the
        data is optical depth. If providing flux, it will be converted to
        apparant optical depth.
    center : :class:`~astropy.units.Quantity`
        The centroid of the ion.
    """
    if continuum is not None:
        y = continuum - y

    mask = (y > 0.001)
    y = y[mask]
    x = x[mask]
    y_max = np.argmax(y)

    if y.size > 0:
        y95 = np.percentile(y, 95)
        y5 = np.percentile(y, 5)

        v95 = x[find_nearest(y[y_max:], y95) + y_max]
        v5 = x[find_nearest(y[:y_max], y5)]

        print(y[y_max], y95, y5, v95, v5)
    else:
        logging.warning("No reasonable amount of optical depth found in "
                        "feature, aborting dv90 calculation.")

        return u.Quantity(0, 'km/s')

    return np.abs((v95 - v5).to('km/s'))


@u.quantity_input(x=['length', 'speed'])
def full_width_half_max(x, y):
    """
    Use a univariate spline to fit the line feature, taking its roots as
    representative of the full width at half maximum.

    Parameters
    ----------
    x : :class:`astropy.units.Quantity`
        The dispersion array.
    y : np.ndarray
        The data array.

    Returns
    -------
    float
        The full width at half maximum.
    """
    # Width can be estimated by the weighted 2nd moment of the x coordinate
    spline = UnivariateSpline(x, y - (np.max(y) + np.min(y)) / 2, s=0)
    r1, r2 = spline.roots()

    # return (r2 - r1) * x.unit
    return _compute_single_fwhm(y, x)


@u.quantity_input(x=['length', 'speed'])
def equivalent_width(x, y):
    fwhm = full_width_half_max(x, y)
    sigma = fwhm / 2.355

    # Amplitude is derived from area
    delta_x = x[1:] - x[:-1]
    sum_y = np.sum(y[1:] * delta_x)
    height = sum_y / (sigma * np.sqrt(2 * np.pi))

    # Calculate equivalent width
    return sum_y / np.max(y)