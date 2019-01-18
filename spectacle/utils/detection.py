import numpy as np
from astropy.convolution import convolve
from spectacle.utils.misc import find_nearest
from scipy.signal import savgol_filter
import astropy.units as u


def region_bounds(x, y, threshold=0, min_distance=1):
    # Slice the y axis in half -- if more data elements exist in the "top"
    # half, assume the spectrum is absorption. Otherwise, assume emission.
    mid_y = min(y) + (max(y) - min(y)) * 0.5
    is_absorption = len(y[y > mid_y]) > len(y[y < mid_y])

    if not isinstance(min_distance, u.Quantity):
        min_distance *= x.unit

    if is_absorption:
        thresh_mask = np.less(y, threshold)
    else:
        thresh_mask = np.greater(y, threshold)

    kernel = [1, 0, -1]

    dY = convolve(y, kernel, 'extend', normalize_kernel=False) #np.diff(y)
    ddY = convolve(dY, kernel, 'extend', normalize_kernel=False) #np.diff(dY)
    dddY = convolve(ddY, kernel, 'extend', normalize_kernel=False) #np.diff(ddY)

    dS = convolve(np.sign(dY), kernel, 'extend', normalize_kernel=False)
    ddS = convolve(np.sign(ddY), kernel, 'extend', normalize_kernel=False)
    dddS = convolve(np.sign(dddY), kernel, 'extend', normalize_kernel=False)

    ddS_mask = (ddS == 0) & thresh_mask

    if is_absorption:
        dS_mask = (dS > 0) & thresh_mask
        dddS_mask = (dddS < 0) & thresh_mask
    else:
        dS_mask = (dS < 0) & thresh_mask
        dddS_mask = (dddS > 0) & thresh_mask

    prime_regions = {}
    ternary_regions = {}

    # Find "buried" lines. Do this by taking the third difference of the
    # spectrum.
    for tind in np.where(dddS_mask)[0][::2]:
        lower_ind = find_nearest(
            np.ma.array(x.value, mask=ddS_mask | (x.value > x.value[tind])),
            x.value[tind])
        upper_ind = find_nearest(
            np.ma.array(x.value, mask=ddS_mask | (x.value < x.value[tind])),
            x.value[tind])

        lower_x_ddS, upper_x_ddS = x.value[lower_ind][0], x.value[upper_ind][0]
        x_dddS = x[tind]

        if np.sign(ddS[lower_ind]) < np.sign(ddS[upper_ind]):
            # Ensure that this found peak value is not within the minimum
            # distance of any other found peak values.
            if np.all([np.abs(x - x_dddS) > min_distance
                       for x, _ in ternary_regions.values()]):
                ternary_regions[(lower_x_ddS, upper_x_ddS)] = (x_dddS, True)

    # Find obvious lines by peak values.
    for pind in np.where(dS_mask)[0][::2]:
        lower_ind = find_nearest(
            np.ma.array(x.value, mask=ddS_mask | (x.value > x.value[pind])),
            x.value[pind])
        upper_ind = find_nearest(
            np.ma.array(x.value, mask=ddS_mask | (x.value < x.value[pind])),
            x.value[pind])

        lower_x_ddS, upper_x_ddS = x.value[lower_ind][0], x.value[upper_ind][0]
        x_dS = x[pind]

        if np.sign(ddS[lower_ind]) < np.sign(ddS[upper_ind]):
            # Ensure that this found peak value is not within the minimum
            # distance of any other found peak values.
            if np.all([np.abs(x - x_dS) > min_distance
                       for x, _ in ternary_regions.values()]):
                prime_regions[(lower_x_ddS, upper_x_ddS)] = (x_dS, False)

    prime_regions.update(ternary_regions)

    return prime_regions