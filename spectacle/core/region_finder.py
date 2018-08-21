import astropy.units as u

import numpy as np
from scipy.signal import savgol_filter
import logging


def find_regions(data, continuum=None, cap_value=None, smooth=False,
                 rel_tol=1e-2, abs_tol=1e-5):
    if cap_value is not None:
        data = np.array(data)
        data[data > cap_value] = cap_value

    if continuum is None:
        continuum = np.zeros(data.shape)

    data = savgol_filter(data, 49, 3) if smooth else data

    # cind = find_nearest(dispersion, center)
    creg = _get_absorption_regions(data, continuum=continuum, rel_tol=rel_tol,
                                   abs_tol=abs_tol)

    return creg


def _get_absorption_regions(data, continuum, rel_tol, abs_tol):
    # This makes the assumption that the continuum has been normalized to 1
    mask = ~np.isclose(data, continuum, rtol=rel_tol, atol=abs_tol)

    creg = contiguous_regions(mask)

    return creg


def contiguous_regions(condition):
    """
    Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index.
    """
    # Find the indices of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    # idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size - 1]

    # Reshape the result into two columns
    idx.shape = (-1, 2)

    return idx