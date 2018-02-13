import logging

import astropy.units as u
from astropy.constants import c, m_e
import numpy as np
from pandas import DataFrame
from scipy.integrate import simps

from ..modeling import VelocityConvert, WavelengthConvert
from ..utils import find_nearest


@u.quantity_input(x=['length', 'speed'], center=['length'])
def dv90(x, y, continuum=None, center=None, ion_name=None):
    equivalencies = [(u.Unit('km/s'), u.Unit('Angstrom'),
                      lambda x: WavelengthConvert(center)(x * u.Unit('km/s')),
                      lambda x: VelocityConvert(center)(x * u.Unit('Angstrom')))]

    def _calculate(x, y, continuum):
        x = x.to(u.Unit('km/s'), equivalencies=equivalencies)

        if continuum is None:
            continuum = 1.0
            logging.info("No continuum provided, assuming 1.0.")

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

        return (v95 - v5).to('km/s')

    return _calculate(x, y, continuum)


@u.quantity_input(x=['length', 'speed'], center=['length'])
def equivalent_width(x, y, continuum=None, center=None, ion_name=None):
    equivalencies = [(u.Unit('km/s'), u.Unit('Angstrom'),
                      lambda x: WavelengthConvert(center)(x * u.Unit('km/s')),
                      lambda x: VelocityConvert(center)(x * u.Unit('Angstrom')))]

    @u.quantity_input(x=u.Unit('km/s'), equivalencies=equivalencies)
    def _calculate(x, y, continuum):
        # Continuum is always assumed to be 1.0
        continuum = continuum if continuum is not None else 1.0

        # Average dispersion in the line region.
        avg_dx = np.mean(x[1:] - x[:-1])

        # Calculate equivalent width
        ew = ((continuum - y / continuum) * avg_dx).sum()

        return ew

    return _calculate(x, y, continuum)
