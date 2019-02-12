import numpy as np
from functools import wraps

import astropy.units as u
import collections

DOPPLER_CONVERT = {
    'optical': u.doppler_optical,
    'radio': u.doppler_radio,
    'relativistic': u.doppler_relativistic
}


def find_nearest(array, value, side="left", count=1):
    """
    The function below works whether or not the input array is sorted. The
    function below returns the index of the input array corresponding to the
    closest value, which is somewhat more general.
    """
    if side == "right":
        indexes = np.abs(array[::-1] - value)
    else:
        indexes = np.abs(array - value)

    if count == 1:
        return np.argsort(indexes)[0]

    return np.argsort(indexes)[:count]


def unit_validator(equivalencies=None, **dwargs):
    def unit_validator_decorator(func):
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            # Validate input units

            return func(*args, **kwargs)
        return func_wrapper
    return unit_validator_decorator


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