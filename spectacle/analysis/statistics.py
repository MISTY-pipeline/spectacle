import logging

import astropy.units as u
from astropy.constants import c, m_e
import numpy as np
from pandas import DataFrame
from scipy.integrate import simps

from ..modeling import VelocityConvert, WavelengthConvert
from ..utils import find_nearest


@u.quantity_input(x=['length', 'speed'], center=['length'])
def delta_v_90(x, y, center=None, continuum=None, ion_name=None):
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
    equivalencies = [(u.Unit('km/s'), u.Unit('Angstrom'),
                      lambda x: WavelengthConvert(center)(x * u.Unit('km/s')),
                      lambda x: VelocityConvert(center)(x * u.Unit('Angstrom')))]

    x = x.to(u.Unit('km/s'), equivalencies=equivalencies)

    if continuum is not None:
        y = continuum - y

    mask = [y > 1e-5]
    y = y[mask]
    x = x[mask]

    if y.size > 0:
        y95 = np.percentile(y, 95, interpolation='midpoint')
        y5 = np.percentile(y, 5, interpolation='midpoint')

        v95 = x[find_nearest(y, y95)]
        v5 = x[find_nearest(y, y5)]
    else:
        logging.warning("No reasonable amount of optical depth found in "
                        "spectrum, aborting dv90 calculation.")

        return u.Quantity(0)

    return np.abs((v95 - v5).to('km/s'))


@u.quantity_input(x=['length', 'speed'])
def equivalent_width(x, y, continuum=None, ion_name=None):
    if continuum is None:
        y = y + 1
        continuum = 1

    # Average dispersion in the line region.
    avg_dx = np.mean(x[1:] - x[:-1])

    # Calculate equivalent width
    ew = ((1 - y / continuum) * avg_dx).sum()

    return np.abs(ew)
