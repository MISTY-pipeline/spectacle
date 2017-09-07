import numpy as np
from pandas import DataFrame


def delta_v_90(x, y):
    df = DataFrame(list(zip(x.value, y)), columns=('x', 'y'))

    x5 = df.quantile(0.25)['x']
    x95 = df.quantile(0.65)['x']
    x50 = df.quantile(0.5)

    return (x95 - x5, x5, x95, x50)


def delta_v_90(x, y):
    tau_tot = quad(lambda x: self(x * u.Unit('Angstrom')),
                   wave_space(velocity[0]).value,
                   wave_space(velocity[-1]).value)

    tau = quad(lambda x: self(x * u.Unit('Angstrom')),
               wave_space(velocity[0]).value,
               wave_space(velocity[-1]).value)

    while tau[0] / tau_tot[0] < 0.9:
        mn -= 1
        mx += 1
        tau = quad(lambda x: self(x * u.Unit('Angstrom')),
                   wave_space(velocity[mn]).value,
                   wave_space(velocity[mx]).value)

    return (velocity[mn], velocity[mx])