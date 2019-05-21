import logging

import astropy.units as u
import numpy as np
from astropy.convolution import convolve

from spectacle.utils.misc import find_nearest
from collections import OrderedDict

__all__ = ['region_bounds']


def _make_data_diffs(y):
    kernel = [1, 0, -1]

    dY = convolve(y, kernel, 'extend', normalize_kernel=False) #np.diff(y)
    ddY = convolve(dY, kernel, 'extend', normalize_kernel=False) #np.diff(dY)
    dddY = convolve(ddY, kernel, 'extend', normalize_kernel=False) #np.diff(ddY)

    return dY, ddY, dddY


def _make_sign_diffs(dY, ddY, dddY):
    kernel = [1, 0, -1]

    # Anywhere that dS >/< 0 is a line for absorption/emission.
    # Anywhere that ddS == 0
    dS = convolve(np.sign(dY), kernel, 'extend', normalize_kernel=False)
    ddS = convolve(np.sign(ddY), kernel, 'extend', normalize_kernel=False)
    dddS = convolve(np.sign(dddY), kernel, 'extend', normalize_kernel=False)

    return dS, ddS, dddS


def _generate_masks(y, threshold, dS, ddS, dddS, is_absorption):
    if is_absorption:
        thresh_mask = np.greater(np.max(y) - y, threshold)
    else:
        thresh_mask = np.greater(y, threshold)

    # Mask areas that don't provide line information. The secondary
    # convolution should gives us the bounds of when we enter (upward slope,
    # for absorption) and exit (downward slope, for absorption) a feature.
    ddS_mask = ((ddS < 0) | (ddS > 0)) & thresh_mask

    # The tertiary convolution provides information on buried lines. If, in
    # absorption, there is a negative convolution value between the bounds
    # defined in the secondary convolution, then there is a buried line.
    # dddS_mask = ((dddS < 0) | (dddS > 0)) & thresh_mask

    if is_absorption:
        # Mask for lines in absorption. During convolution, the values of ds
        # that are < 0 represent downward-curved changes in the spectrum
        # (i.e. when the we a peak close to the continuum between peaks). So,
        # only grab the parts where we have a upward curve (i.e. when we've
        # hit the bottom of an absorption trough).
        dS_mask = (dS > 0) & thresh_mask
        dddS_mask = (dddS < 0) & thresh_mask
    else:
        # Mask for lines in emission
        dS_mask = (dS < 0) & thresh_mask
        dddS_mask = (dddS > 0) & thresh_mask

    return dS_mask, ddS_mask, dddS_mask


def _find_ternary_bounds(x, ddS_mask, dddS_mask, min_distance, is_absorption):
    ternary_regions = OrderedDict()

    # Find "buried" lines. Do this by taking the third difference of the
    # spectrum. Find the indices where the dddS_mask is true, retrieve only a
    # single index from the tuple by going in steps of 2. Each tind represents
    # the index of the centroid of the found buried line.
    for tind in np.where(dddS_mask)[0][::2]:
        # ddS contains bounds information. Find the lower bound index of the
        # dispersion.
        lower_ind = find_nearest(
            np.ma.array(x.value, mask=~ddS_mask | (x.value > x.value[tind])),
            x.value[tind])
        # ddS contains bounds information. Find the upper bound index of the
        # dispersion.
        upper_ind = find_nearest(
            np.ma.array(x.value, mask=~ddS_mask | (x.value < x.value[tind])),
            x.value[tind])

        # Retrieve the dispersion value for these indices and set the
        # dispersion value for the centroid.
        lower_x_ddS, upper_x_ddS = x.value[lower_ind], \
                                     x.value[upper_ind]
        x_dddS = x[tind]

        # if is_absorption:
        #     # This is truly a buried line if, for absorption, if the sign of
        #     # the lower index in the bounds mask is greater than the upper.
        #     cond = (dddS[lower_ind] > dddS[upper_ind])
        # else:
        #     # This is truly a buried line if, for emission, if the sign of
        #     # the lower index in the bounds mask is greater than the upper.
        #     cond = (dddS[lower_ind] < dddS[upper_ind])

        # if True:
        # Ensure that this found peak value is not within the minimum
        # distance of any other found peak values.
        if np.all([np.abs(x - x_dddS) > min_distance
                    for x, _, _, _ in ternary_regions.values()]):
            ternary_regions[(lower_ind, upper_ind)] = (
            x_dddS, (lower_x_ddS, upper_x_ddS), is_absorption, True)

    return ternary_regions


def _find_primary_bounds(x, dS_mask, ddS_mask, min_distance, is_absorption):
    prime_regions = OrderedDict()

    # Find obvious lines by peak values.
    for pind in np.where(dS_mask)[0][::2]:
        lower_ind = find_nearest(
            np.ma.array(x.value, mask=~ddS_mask | (x.value > x.value[pind])),
            x.value[pind])
        upper_ind = find_nearest(
            np.ma.array(x.value, mask=~ddS_mask | (x.value < x.value[pind])),
            x.value[pind])

        lower_x_ddS, upper_x_ddS = x.value[lower_ind], \
                                   x.value[upper_ind]
        x_dS = x[pind]

        # if is_absorption:
        #     cond = np.sign(ddS[lower_ind]) > np.sign(ddS[upper_ind])
        # else:
        #     cond = np.sign(ddS[lower_ind]) < np.sign(ddS[upper_ind])

        # if cond:
        # Ensure that this found peak value is not within the minimum
        # distance of any other found peak values.
        if np.all([np.abs(x - x_dS) > min_distance
                    for x, _, _, _ in prime_regions.values()]):
            prime_regions[(lower_ind, upper_ind)] = (
            x_dS, (lower_x_ddS, upper_x_ddS), is_absorption, False)

    return prime_regions


def region_bounds(x, y, threshold=0.001, min_distance=1):
    # Slice the y axis in half -- if more data elements exist in the "top"
    # half, assume the spectrum is absorption. Otherwise, assume emission.
    mid_y = min(y) + (max(y) - min(y)) * 0.5
    is_absorption = len(y[y > mid_y]) > len(y[y < mid_y])

    if not isinstance(min_distance, u.Quantity):
        min_distance *= x.unit

    dY, ddY, dddY = _make_data_diffs(y)
    dS, ddS, dddS = _make_sign_diffs(dY, ddY, dddY)

    dS_mask, ddS_mask, dddS_mask = _generate_masks(y, threshold, dS, ddS, dddS,
                                                   is_absorption)

    ternary_regions = _find_ternary_bounds(x, ddS_mask, dddS_mask,
                                           min_distance, is_absorption)
    prime_regions = _find_primary_bounds(x, dS_mask, ddS_mask, min_distance,
                                         is_absorption)

    # Delete any information in ternary that shares the same bounds in primary
    for k in prime_regions:
        if k in ternary_regions:
            del ternary_regions[k]

    # For the ternary centroids closest to a prime centroid, use the bounds
    # information of the primary centroid
    for (t_low_ind, t_up_ind), (t_cent, bnds, is_absorb, buried) in ternary_regions.copy().items():
        p_cents = [pr[0] for pr in prime_regions.values()]
        p_bnds = [(lo, hi) for lo, hi in prime_regions]

        # Find closest index
        ind = find_nearest(np.array([c.value for c in p_cents]), t_cent.value)
        p_cent = p_cents[ind]
        p_bnd = p_bnds[ind]
        # p_ind = find_nearest(x.value, p_cent.value)

        new_t_up_ind = t_up_ind
        new_t_low_ind = t_low_ind

        # If the prime centroid is greater than the ternary centroid,
        # update the ternary upper bound.
        if p_cent > t_cent:
            new_t_up_ind = p_bnd[1] # p_ind
        else:
            new_t_low_ind = p_bnd[0] # p_ind

        del ternary_regions[(t_low_ind, t_up_ind)]

        ternary_regions[(new_t_low_ind, new_t_up_ind)] = (
            t_cent, (x.value[new_t_low_ind], x.value[new_t_up_ind]),
            is_absorb, buried)

    ternary_regions.update(prime_regions)

    return ternary_regions