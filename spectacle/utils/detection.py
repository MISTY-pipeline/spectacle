import numpy as np
from astropy.convolution import convolve
from spectacle.utils.misc import find_nearest
from scipy.signal import savgol_filter


def region_bounds(x, y, threshold=0):
    # Slice the y axis in half -- if more data elements exist in the "top"
    # half, assume the spectrum is absorption. Otherwise, assume emission.
    mid_y = min(y) + (max(y) - min(y)) * 0.5
    is_absorption = len(y[y > mid_y]) > len(y[y < mid_y])

    print(mid_y, len(y[y > mid_y]), len(y[y < mid_y]))

    if is_absorption:
        thresh_mask = np.less(y, threshold)
    else:
        thresh_mask = np.greater(y, threshold)

    kernel = [1, 0, -1]

    window = y.size/100
    window = window + 1 if window % 2 == 0 else window
    window = 501

    y = savgol_filter(y, window , 2)
    dY = convolve(y, kernel, 'extend', normalize_kernel=False) #np.diff(y)
    dY = savgol_filter(dY, window, 2)
    ddY = convolve(dY, kernel, 'extend', normalize_kernel=False) #np.diff(dY)
    ddY = savgol_filter(ddY, window, 2)
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

    regions = {}

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
            regions[(lower_x_ddS, upper_x_ddS)] = (x_dddS, True)

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
            regions[(lower_x_ddS, upper_x_ddS)] = (x_dS, False)

    return regions