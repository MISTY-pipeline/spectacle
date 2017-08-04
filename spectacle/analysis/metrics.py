import logging
import numpy as np
import uncertainties.unumpy as unp
from scipy import stats
from functools import wraps
from ..analysis.resample import resample


def _format_arrays(a, v, use_tau=False, masking=True, resamp_bins=False):
    # Extract only the parts of the spectrum with data in it
    a_mask = a.line_mask
    v_mask = v.line_mask

    # a_region_mask = np.ones(a.data.shape, dtype=bool)
    # v_region_mask = np.ones(v.data.shape, dtype=bool)
    mask = (a_mask | v_mask)

    # Apply masks
    al = a.data[a_mask]
    vl = v.data[v_mask]

    al_disp = a.dispersion[a_mask]
    vl_disp = v.dispersion[v_mask]

    # min_val = al_disp[0] if al_disp[a_mask][0] < vl_disp[0] else vl_disp[0]
    # max_val = al_disp[-1] if al_disp[-1] < vl_disp[-1] else vl_disp[-1]

    masker = lambda x: x[mask]

    # Check for consistency
    d_al = a.dispersion[1] - a.dispersion[0]
    d_vl = v.dispersion[1] - v.dispersion[0]

    # If the two spectra are not the same dimension, resample to lower
    # resolution
    if d_al != d_vl:
        logging.warning("Dispersions have different deltas: {} and {}. "
                        "Resampling to smallest delta.".format(d_al, d_vl))

    if d_al > d_vl:
        a = a.resample(v.dispersion)
    elif d_vl > d_al:
        v = v.resample(a.dispersion)

    mask = None if not masking else mask

    # Compose the uncertainty arrays
    if use_tau:
        al, vl = unp.uarray(masker(a.tau), masker(a.tau_uncertainty)), \
                 unp.uarray(masker(v.tau), masker(v.tau_uncertainty))
    else:
        al, vl = unp.uarray(masker(a.data), masker(a.uncertainty)), \
                 unp.uarray(masker(v.data), masker(v.uncertainty))

    if resamp_bins:
        if al.size > vl.size:
            remat = resample(vl_disp,
                             np.linspace(vl_disp[0], vl_disp[-1], al.size))
            vl = np.dot(remat, vl)
        elif vl.size > al.size:
            remat = resample(al_disp,
                             np.linspace(al_disp[0], al_disp[-1], vl.size))
            al = np.dot(remat, al)

    return al, vl, mask



def metric(strip=True):
    def decorator(func):
        @wraps(func)
        def func_wrapper(a, v, use_tau=False, masking=True, *args, **kwargs):
            al, vl, mask = _format_arrays(a, v, use_tau=use_tau,
                                          masking=masking)

            if strip:
                al, vl = unp.nominal_values(al), unp.nominal_values(vl)

                return func(al, vl, *args, **kwargs), np.zeros(al.shape), mask

            res = func(al, vl, *args, **kwargs)

            return unp.nominal_values(res), unp.std_devs(res), mask

        return func_wrapper
    return decorator


@metric()
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
    if normalize:
        a = (a - a.mean()) / (np.std(a) * a.size)
        v = (v - v.mean()) / np.std(v)

    return np.sum(np.correlate(a, v, mode))


@metric(strip=False)
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
    return (a[:-1] * a[1:]).mean() / a.mean() ** 2


@metric(strip=True)
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
    return np.corrcoef(a, v)[0, 1]


@metric(strip=False)
def correlate(a, v, mode='true'):
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
    if mode == 'true':
        return np.sum((a * v) / (unp.sqrt((a ** 2).sum()) *
                                 unp.sqrt((v ** 2).sum())))
    elif mode == 'full':
        sh_vl = np.random.permutation(v)
        sh_al = np.random.permutation(a)
        return (a - v) ** 2 / (sh_al * sh_vl)
    elif mode == 'lite':
        sh_vl = np.random.permutation(v)
        sh_al = np.random.permutation(a)
        return (a * v) / (sh_al * sh_vl)
    else:
        raise NameError("No such mode: {}".format(mode))


@metric()
def anderson_darling(a, v, *args, **kwargs):
    a2, crit, p = stats.anderson_ksamp([a, v], *args, **kwargs)

    return a2


@metric()
def kendalls_tau(a, v, *args, **kwargs):
    corr, p = stats.kendalltau(a, v, *args, **kwargs)

    return corr


@metric()
def kolmogorov_smirnov(a, v, *args, **kwargs):
    corr, p = stats.ks_2samp(a, v)

    return corr
