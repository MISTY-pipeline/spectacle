import logging
import numpy as np
# from skimage.feature import match_template
import uncertainties.unumpy as unp


def _format_arrays(a, v):
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

    al, vl = unp.uarray(a.flux[mask], a.uncertainty[mask]), \
             unp.uarray(v.flux[mask], v.uncertainty[mask])

    d_al = al[1].nominal_value - al[0].nominal_value
    d_vl = vl[1].nominal_value - vl[0].nominal_value

    # If the two spectra are not the same dimension, resample to lower
    # resolution
    if d_al != d_vl:
        logging.warning("Dispersions have different deltas: {} and {}. "
                        "Resampling to smallest delta.".format(d_al, d_vl))

    if d_al > d_vl:
        a = a.resample(v.dispersion)
        al = unp.uarray(a.flux[mask], a.uncertainty[mask])
    elif d_vl > d_al:
        v = v.resample(a.dispersion)
        vl = unp.uarray(v.flux[mask], v.uncertainty[mask])

    return al, vl


def npcorrelate(a, v, mode='valid', normalize=False):
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
    al, vl = _format_arrays(a, v)

    if normalize:
        al = (al - al.mean()) / (al.std_dev() * al.size)
        vl = (vl - vl.mean()) / vl.std_dev()

    ret = np.correlate(al, vl, mode)

    return unp.nominal_values(ret), unp.std_devs(ret)


def autocorrelate(a):
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
    af = unp.uarray(a.flux, a.uncertainty)

    fin = unp.uarray(np.zeros(a.flux.size), np.zeros(a.flux.size))

    for dv in range(fin.size):
        for i in range(fin.size):
            fin[dv] += af[dv] * af[i]

    ret = np.mean(fin)/(np.mean(fin) ** 2)

    return ret.nominal_value, ret.std_dev


def cross_correlate(a, v):
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
    al, vl = _format_arrays(a, v)

    mat = np.corrcoef(al, vl)[0, 1]

    return unp.nominal_values(mat), unp.std_devs(mat)


def correlate(a, v, mode='full'):
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
    al, vl = _format_arrays(a, v)

    # al = al[np.isfinite(al)]
    # vl = vl[np.isfinite(vl)]

    sh_vl = np.random.permutation(vl)
    sh_al = np.random.permutation(al)

    if mode == 'full':
        ret = (al - vl) ** 2 / (sh_al * sh_vl) ** 2
    elif mode == 'lite':
        ret = (al * vl) / (sh_al * sh_vl)
    else:
        raise NameError("No such mode: {}".format(mode))

    return unp.nominal_values(ret), unp.std_devs(ret)
