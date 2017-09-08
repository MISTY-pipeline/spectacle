import numpy as np
from pandas import DataFrame
from scipy.integrate import simps
import astropy.units as u

from ..models import WavelengthConvert, VelocityConvert

# def delta_v_90(x, y):
#     df = DataFrame(list(zip(x.value, y)), columns=('x', 'y'))
#
#     x5 = df.quantile(0.25)['x']
#     x95 = df.quantile(0.65)['x']
#     x50 = df.quantile(0.5)
#
#     return (x95 - x5, x5, x95, x50)


@u.quantity_input(center=u.Unit('Angstrom'))
def delta_v_90(x, y, center=None):

    if isinstance(x, np.ma.MaskedArray):
        x = x.compressed()

    if isinstance(y, np.ma.MaskedArray):
        y = y.compressed()

    equivalencies = [(u.Unit('km/s'), u.Unit('Angstrom'),
                      lambda x: WavelengthConvert(center)(x * u.Unit('km/s')),
                      lambda x: VelocityConvert(center)(x * u.Unit('Angstrom')))]

    @u.quantity_input(x=u.Unit('km/s'), equivalencies=equivalencies)
    def _calculate(x, y):
        # tau_tot = simps(y, x.value)
        # tau = simps(y, x.value)

        mn, mx = 0, -1

        # while tau / tau_tot > 0.9:
        tau_tot = tau_left = simps(y[mn:mx], x.value[mn:mx])

        # Take 0.05tau off left
        while tau_left / tau_tot > 0.95:
            mn += 1
            tau_left = simps(y[mn:mx], x.value[mn:mx])

        tau_tot = tau_right = simps(y[mn:mx], x.value[mn:mx])

        # Take 0.05 tau off right
        while tau_right / tau_tot > 0.95:
            mx -= 1
            tau_right = simps(y[mn:mx], x.value[mn:mx])

        # tau = simps(y[mn:mx], x.value[mn:mx])

        return x[mn], x[mx]

    return _calculate(x, y)