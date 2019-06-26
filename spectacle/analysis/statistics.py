import astropy.units as u
import numpy as np
import logging
from ..utils.misc import find_nearest
from specutils.analysis.width import _compute_single_fwhm
from scipy.interpolate import UnivariateSpline
from astropy.modeling import FittableModel

__all__ = ['delta_v_90', 'full_width_half_max', 'equivalent_width']


@u.quantity_input(x=['length', 'speed'])
def delta_v_90(x, y):
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
    """
    tot_tau = np.sum(y)#, x)

    lower_ind = 0
    less_five = np.sum(y[lower_ind:])

    while (less_five/tot_tau) > 0.95:
        less_five = np.sum(y[lower_ind:])#, x[lower_ind:])
        lower_ind += 1

    upper_ind = -1
    less_five = np.sum(y[:upper_ind])

    while (less_five/tot_tau) > 0.95:
        less_five = np.sum(y[:upper_ind])#, x[:upper_ind])
        upper_ind -= 1

    return np.abs((x[upper_ind] - x[lower_ind]).to('km/s'))


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
    r = spline.roots()

    return (r[-1] - r[0]) * x.unit
    # return _compute_single_fwhm(y, x)


@u.quantity_input(x=['length', 'speed'])
def equivalent_width(x, y, continuum=None):
    norm_y = y.copy()

    if continuum is not None:
        if issubclass(type(continuum), FittableModel):
            continuum = continuum(x)

        ew = np.trapz((continuum - norm_y)/continuum, x=x)
    else:
        ew = np.trapz(norm_y, x=x)

    return np.abs(ew)
