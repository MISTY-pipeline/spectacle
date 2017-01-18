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


def find_bounds(dispersion, data, center, value, cap=False):
    if cap:
        data = np.array(data)
        data[data > value] = value

    cind = find_nearest(dispersion, center)
    creg = _get_absorption_regions(data)

    try:
        ind = np.where((creg[:, 0] <= cind) & (creg[:, 1] >= cind))[0][0]
    except IndexError:
        return 0, len(data) - 1

    left_ind, right_ind = creg[ind]

    return left_ind, right_ind


def _get_absorption_regions(data):
    # This makes the assumption that the continuum has been normalized to 1
    cont = np.ones(data.shape)
    mask = ~np.isclose(data, cont, rtol=1e-2, atol=1e-5)

    creg = contiguous_regions(mask)

    return creg


def contiguous_regions(condition):
    """
    Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index.
    """

    # Find the indicies of changes in "condition"
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