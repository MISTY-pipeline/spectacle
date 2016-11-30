import logging
import numpy as np
# from skimage.feature import match_template
import uncertainties.unumpy as unp


def _format_arrays(a, v, use_tau=False, use_region=True):
    # Extract only the parts of the spectrum with data in it
    a_region_mask = None
    v_region_mask = None

    if use_region:
        a_region_mask = a._get_range_mask()
        v_region_mask = v._get_range_mask()

        diff = a_region_mask[a_region_mask].size - v_region_mask[v_region_mask].size

        if diff > 0:
            diff = np.abs(diff)
            min_add, max_add = diff // 2, diff // 2 + diff % 2
            mn_ind, mx_ind = np.argmax(v_region_mask), \
                             v_region_mask.size - np.argmax(v_region_mask[::-1]) - 1
            v_region_mask[mn_ind-min_add:mx_ind+1+max_add] = True
        else:
            diff = np.abs(diff)
            min_add, max_add = diff // 2, diff // 2 + diff % 2
            mn_ind, mx_ind = np.argmax(a_region_mask), \
                             a_region_mask.size - np.argmax(a_region_mask[::-1]) - 1
            a_region_mask[mn_ind-min_add:mx_ind+1+max_add] = True

    # Clip the spectra to the same range
    if a.dispersion[0] > v.dispersion[0]:
        min_mask = (v.dispersion >= a.dispersion[0])
    else:
        min_mask = (a.dispersion >= v.dispersion[0])

    if a.dispersion[-1] < v.dispersion[-1]:
        max_mask = (v.dispersion <= a.dispersion[-1])
    else:
        max_mask = (a.dispersion <= v.dispersion[-1])

    mask = min_mask & max_mask
    a_mask = mask & a_region_mask
    v_mask = mask & v_region_mask

    # Compose the uncertainty arrays
    if use_tau:
        al, vl = unp.uarray(a.tau[a_mask], a.tau_uncertainty[a_mask]), \
                 unp.uarray(v.tau[v_mask], v.tau_uncertainty[v_mask])
    else:
        al, vl = unp.uarray(a.flux[a_mask], a.uncertainty[a_mask]), \
                 unp.uarray(v.flux[v_mask], v.uncertainty[v_mask])

    d_al = a.dispersion[1] - a.dispersion[0]
    d_vl = v.dispersion[1] - v.dispersion[0]

    # If the two spectra are not the same dimension, resample to lower
    # resolution
    if d_al != d_vl:
        logging.warning("Dispersions have different deltas: {} and {}. "
                        "Resampling to smallest delta.".format(d_al, d_vl))

    if d_al > d_vl:
        a = a.resample(v.dispersion)

        if use_tau:
            al = unp.uarray(a.tau[a_mask], a.tau_uncertainty[a_mask])
        else:
            al = unp.uarray(a.flux[a_mask], a.uncertainty[a_mask])
    elif d_vl > d_al:
        v = v.resample(a.dispersion)

        if use_tau:
            vl = unp.uarray(v.tau[v_mask], v.tau_uncertainty[v_mask])
        else:
            vl = unp.uarray(v.flux[v_mask], v.uncertainty[v_mask])

    return al, vl


def npcorrelate(a, v, mode='valid', normalize=False, use_tau=False):
    """
    Returns the 1d cross-correlation of two input arrays.

    Parameters
    ----------
    a : ndarray
        First 1d array.
    v : ndarray
        Second 1d array.
    normalize : bool
        Should the result be normalized?

    Returns
    -------
    <returned value> : float
        The (normalized) correlation.
    """
    al, vl = _format_arrays(a, v, use_tau=use_tau)

    if normalize:
        al = (al - al.mean()) / (al.std_dev() * al.size)
        vl = (vl - vl.mean()) / vl.std_dev()

    ret = np.correlate(al, vl, mode)

    return unp.nominal_values(ret), unp.std_devs(ret)


def autocorrelate(a, use_tau=False):
    """
    Implementation of the correlation function in Peeples et al. 2010.

    Parameters
    ----------
    a : :class:spectra.ProfileSpectrum
        Spectrum to correlate with itself.

    Returns
    -------
    ret : float
        Value representing the correlation.
    """
    if use_tau:
        af = unp.uarray(a.tau, a.tau_uncertainty)
    else:
        af = unp.uarray(a.flux, a.uncertainty)

    # fin = unp.uarray(np.zeros(a.flux.size), np.zeros(a.flux.size))
    #
    # for dv in range(fin.size):
    #     for i in range(fin.size):
    #         fin[dv] += af[dv] * af[i]

    ret = (af[:-1] * af[1:]).mean() / af.mean() ** 2

    # ret = np.mean(fin, axis=0)/(np.mean(fin, axis=0) ** 2)

    return ret.nominal_value, ret.std_dev


def cross_correlate(a, v, use_tau=False):
    """
    Calculates the Pearson product-moment correlation. Returns the
    normalized covariance matrix.

    Parameters
    ----------
    a : spectra.ProfileSpectrum
        First spectrum.
    v : spectra.ProfileSpectrum
        Second spectrum.

    Returns
    -------
    mat : ndarray
        The correlation coefficient matrix of the variables.
    """
    al, vl = _format_arrays(a, v, use_tau=use_tau)

    mat = np.corrcoef(unp.nominal_values(al), unp.nominal_values(vl))[0, 1]

    return mat


def correlate(a, v, mode='full', use_tau=False):
    """
    Correlation function described by Molly Peeples.

    Parameters
    ----------
    a : :class:spectra.ProfileSpectrum
        First spectrum.
    v : :class:spectra.ProfileSpectrum
        Second spectrum.
    mode : {'full', 'lite'}, optional
        Which of the two modes to use when calculating correlation.

        * 'full': ``(D_1 - D_2) ** 2 / (R_1 * R_2) ** 2``
        * 'lite': ``(D_1 * D_2) / (R_1 * R_2) ** 2``

    Returns
    -------
     ret : ndarray
        An array describing the correlation at every position.
    """
    al, vl = _format_arrays(a, v, use_tau=use_tau)

    # al = al[np.isfinite(al)]
    # vl = vl[np.isfinite(vl)]

    sh_vl = np.random.permutation(vl)
    sh_al = np.random.permutation(al)

    if mode == 'full':
        ret = (al - vl) ** 2 / (sh_al * sh_vl)
    elif mode == 'lite':
        ret = (al * vl) / (sh_al * sh_vl)
    else:
        raise NameError("No such mode: {}".format(mode))

    return unp.nominal_values(ret), unp.std_devs(ret)
