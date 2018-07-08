import numpy as np
from functools import wraps

import astropy.units as u
import collections


def find_nearest(array, value, side="left"):
    """
    The function below works whether or not the input array is sorted. The
    function below returns the index of the input array corresponding to the
    closest value, which is somewhat more general.
    """
    if side == "right":
        idx = (np.abs(array[::-1] - value)).argmin()
    else:
        idx = (np.abs(array - value)).argmin()

    return idx


def unit_validator(equivalencies=None, **dwargs):
    def unit_validator_decorator(func):
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            # Validate input units

            return func(*args, **kwargs)
        return func_wrapper
    return unit_validator_decorator


def wave_to_vel_equiv(center):
    from ..modeling.converters import WavelengthConvert, VelocityConvert

    return [(u.Unit('km/s'),
             u.Unit('Angstrom'),
             lambda x: WavelengthConvert(center)(x * u.Unit('km/s')),
             lambda x: VelocityConvert(center)(x * u.Unit('Angstrom')))]


def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
