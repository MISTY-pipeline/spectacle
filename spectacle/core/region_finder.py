import peakutils
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
        continuum = np.ones(data.shape)

    data = savgol_filter(data, 49, 3) if smooth else data

    # cind = find_nearest(dispersion, center)
    creg = _get_absorption_regions(data, continuum=continuum, rel_tol=rel_tol,
                                   abs_tol=abs_tol)

    # Find the center point of all the regions
    avg_list = list(map(lambda i: int(i[0] + (i[1] - i[0]) * 0.5), creg[:, :]))

    if len(avg_list) == 0:
        logging.warning("No absorption regions identified; defaulting to "
                        "entire bounds of spectrum.")
        return [(0, len(data) - 1)]

    # cind = avg_list[find_nearest(avg_list, cind)]

    cont_reg = []

    for cind in avg_list:
        # For the closest center to the provided lambda_0 value, see if there
        # is an identified region for it
        ind = np.where((creg[:, 0] <= cind) & (creg[:, 1] >= cind))
        ind = ind[0][0]

        left_ind, right_ind = creg[ind]

        if (right_ind - 1 - left_ind) <= 1:
            logging.error(
                "Improper boundaries found; defaulting to entire range.")
            continue

        # Add left and right indices of contiguous region to list
        cont_reg.append((left_ind, right_ind - 1))

    return cont_reg


def _get_absorption_regions(data, continuum, rel_tol, abs_tol):
    # This makes the assumption that the continuum has been normalized to 1
    mask = ~np.isclose(continuum, data, rtol=rel_tol, atol=abs_tol)

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
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]

    # Reshape the result into two columns
    idx.shape = (-1, 2)

    return idx