import numpy as np
from astropy.convolution import convolve
from spectacle.utils.misc import find_nearest


def region_bounds(x, y):
    # Slice the y axis in half -- if more data elements exist in the "top"
    # half, assume the spectrum is absorption. Otherwise, assume emission.
    mid_y = min(y) + (max(y) - min(y)) * 0.5

    print(mid_y, len(y[y > mid_y]), len(y[y < mid_y]))

    # if len(y[y > mid_y]) > len(y[y < mid_y]):
    #     # Absorption
    #
    #     y = max(y) - y

    kernel = [1, 0, -1]

    dY = convolve(y, kernel, 'extend', normalize_kernel=False) #np.diff(y)
    ddY = convolve(dY, kernel, 'extend', normalize_kernel=False) #np.diff(dY)
    dddY = convolve(ddY, kernel, 'extend', normalize_kernel=False) #np.diff(ddY)

    dS = convolve(np.sign(dY), kernel, 'extend', normalize_kernel=False)
    ddS = convolve(np.sign(ddY), kernel, 'extend', normalize_kernel=False)
    dddS = convolve(np.sign(dddY), kernel, 'extend', normalize_kernel=False)

    regions = {}

    for tind in np.where(dddS > 0)[0][::2]:
        lower_ind = find_nearest(
            np.ma.array(x.value, mask=(ddS == 0) | (x.value > x.value[tind])),
            x.value[tind])
        upper_ind = find_nearest(
            np.ma.array(x.value, mask=(ddS == 0) | (x.value < x.value[tind])),
            x.value[tind])

        lower_x_ddS, upper_x_ddS = x.value[lower_ind][0], x.value[upper_ind][0]
        x_dddS = x[tind]

        if np.sign(ddS[lower_ind]) < np.sign(ddS[upper_ind]):
            regions[(lower_x_ddS, upper_x_ddS)] = (x_dddS, True)

    for pind in np.where(dS < 0)[0][::2]:
        lower_ind = find_nearest(
            np.ma.array(x.value, mask=(ddS == 0) | (x.value > x.value[pind])),
            x.value[pind])
        upper_ind = find_nearest(
            np.ma.array(x.value, mask=(ddS == 0) | (x.value < x.value[pind])),
            x.value[pind])

        lower_x_ddS, upper_x_ddS = x.value[lower_ind][0], x.value[upper_ind][0]
        x_dS = x[pind]

        if np.sign(ddS[lower_ind]) < np.sign(ddS[upper_ind]):
            regions[(lower_x_ddS, upper_x_ddS)] = (x_dS, False)

    return regions