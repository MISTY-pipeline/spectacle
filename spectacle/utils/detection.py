import numpy as np


def region_bounds(x, y):
    from astropy.convolution import convolve
    from spectacle.utils.misc import find_nearest

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