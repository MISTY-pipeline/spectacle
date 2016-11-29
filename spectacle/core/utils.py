import os
import numpy as np
from astropy.table import Table


def find_index(array, value):
    """
    The function below works whether or not the input array is sorted. The
    function below returns the index of the input array corresponding to the
    closest value, which is somewhat more general.
    """
    array = np.array(array)
    idx_sorted = np.argsort(array)
    sorted_array = np.array(array[idx_sorted])
    idx = np.searchsorted(sorted_array, value, side="left")
        
    if idx >= len(array):
        idx_nearest = idx_sorted[len(array)-1]
    elif idx == 0:
        idx_nearest = idx_sorted[0]
    else:
        if abs(value - sorted_array[idx-1]) < abs(value - sorted_array[idx]):
            idx_nearest = idx_sorted[idx-1]
        else:
            idx_nearest = idx_sorted[idx]

    return idx_nearest


ION_TABLE = Table.read(
    os.path.abspath(
        os.path.join(__file__, '..', '..', 'data', 'line_list', 'ions.ecsv')),
    format='ascii.ecsv')
