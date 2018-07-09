import logging

import astropy.units as u
from astropy.constants import c, m_e
import numpy as np
from pandas import DataFrame
from scipy.integrate import simps

from ..utils import find_nearest, wave_to_vel_equiv

dop_rel_equiv = u.equivalencies.doppler_relativistic


@u.quantity_input(x=['length', 'speed'])
def delta_v_90(x, y, continuum=None, rest_wavelength=None):
    """
    Calculate the dispersion that encompasses the central 90 percent of the
    apparant optical depth. Follows the formulation defined in Prochaska &
    Wolf (1997).
    Parameters
    ----------
    x : :class:~`astropy.units.Quantity`
        The dispersion axis. Can be either wavelength or velocity space.
    y : array-like
        Flux or optical depth array. Note that the calculation assumes that the
        data is optical depth. If providing flux, it will be converted to
        apparant optical depth.
    center : :class:~`astropy.units.Quantity`
        The centroid of the ion.
    """
    if continuum is not None:
        y = continuum - y

    if rest_wavelength is not None:
        x = x.to('km/s', equivalencies=dop_rel_equiv(rest_wavelength))

    mask = (y > 0.001)
    y = y[mask]
    x = x[mask]

    if y.size > 0:
        y95 = np.percentile(y, 95)
        y5 = np.percentile(y, 5)

        v95 = x[find_nearest(y, y95)]
        v5 = x[find_nearest(y, y5)]
    else:
        logging.warning("No reasonable amount of optical depth found in "
                        "feature, aborting dv90 calculation.")

        return u.Quantity(0, 'km/s')

    return np.abs((v95 - v5).to('km/s'))


@u.quantity_input(x=['length', 'speed'])
def equivalent_width(x, y, continuum=None):
    if continuum is not None:
        y = continuum - y

    # Average dispersion in the line region.
    avg_dx = np.mean(x[1:] - x[:-1])

    # Calculate equivalent width
    return np.abs(((1 - y) * avg_dx).sum())
