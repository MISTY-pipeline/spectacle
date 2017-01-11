import os
import numpy as np
from astropy.table import Table


def find_nearest(array, value, side="left"):
    """
    The function below works whether or not the input array is sorted. The
    function below returns the index of the input array corresponding to the
    closest value, which is somewhat more general.
    """
    if side == "right":
        idx = len(array) - 1 - (np.abs(array[::-1]-value)).argmin()
    else:
        idx = (np.abs(array-value)).argmin()

    return idx


def find_bounds(array, start_index, value, cap=False):
    if cap:
        array = np.array(array)
        array[array > value] = value

    left_ind = find_nearest(array[:start_index:-1], value, side="right")
    right_ind = start_index + find_nearest(array[start_index:], value)

    return left_ind, right_ind


ION_TABLE = Table.read(
    os.path.abspath(
        os.path.join(__file__, '..', '..', 'data', 'line_list', 'ions.ecsv')),
    format='ascii.ecsv')
