import numpy as np


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