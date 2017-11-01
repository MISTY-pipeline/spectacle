import numpy as np
from functools import wraps



def find_nearest(array, value, side="left"):
    """
    The function below works whether or not the input array is sorted. The
    function below returns the index of the input array corresponding to the
    closest value, which is somewhat more general.
    """
    if side == "right":
        idx = (np.abs(array[::-1]-value)).argmin()
    else:
        idx = (np.abs(array-value)).argmin()

    return idx


def unit_validator(equivalencies=None, **dwargs):
    def unit_validator_decorator(func):
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            # Validate input units


            return func(*args, **kwargs)
        return func_wrapper
    return unit_validator_decorator