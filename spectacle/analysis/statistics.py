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

        tau_tot = tau_left = tau_right = simps(y, x.value)

        # Take 0.05 tau off right
        rmx = -1
        while tau_right / tau_tot > 0.95:
            rmx -= 1
            tau_right = simps(y[:rmx], x.value[:rmx])

        # Take 0.05tau off left
        lmn = 0
        while tau_left / tau_tot > 0.95:
            lmn += 1
            tau_left = simps(y[lmn:], x.value[lmn:])

        return x[lmn], x[rmx]

    return _calculate(x, y)