import logging
import numpy as np
# from skimage.feature import match_template
import uncertainties.unumpy as unp


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
    al, vl = unp.uarray(a.flux, a.uncertainty), \
             unp.uarray(v.flux, v.uncertainty)

    if normalize:
        al = (al - al.mean()) / (al.std_dev() * al.size)
        vl = (vl - vl.mean()) / vl.std_dev()

    return np.correlate(al.nominal_value, vl.nominal_value, mode)


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

    fin = np.zeros(a.flux.size)

    for dv in range(fin.size):
        for i in range(fin.size):
            fin[dv] += af[dv] * af[i]

    ret = np.mean(fin)/(np.mean(fin) ** 2)

    return ret


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
    al, vl = unp.uarray(a.flux, a.uncertainty), \
             unp.uarray(v.flux, v.uncertainty)

    d_al = al[1] - al[0]
    d_vl = vl[1] - vl[0]

    # If the two spectra are not the same dimension, resample to lower
    # resolution
    if d_al != d_vl:
        logging.warning("Arguments have different dimensions: {} and {}. "
                     "Resampling to lowest dimension.".format(al.shape,
                                                              vl.shape))

    if d_al > d_vl:
        al = a.resample(v.dispersion, copy=True)
    elif d_vl > d_al:
        vl = v.resample(a.dispersion, copy=True)

    mat = np.corrcoef(al, vl)[0, 1]
    return mat


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
    al, vl = a.flux, v.flux

    # If the two spectra are not the same dimension, resample to lower
    # resolution
    if al.size != vl.size:
        logging.warning("Arguments have different dimensions: {} and {}. "
                        "Resampling to lowest dimension.".format(al.shape,
                                                                 vl.shape))

    if al.size > vl.size:
        al = a.resample(v.dispersion, copy=True)
    elif vl.size > al.size:
        vl = v.resample(a.dispersion, copy=True)

    print(al.size, vl.size)
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

    return ret
