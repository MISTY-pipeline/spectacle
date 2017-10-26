import abc
import six

from scipy import stats
import numpy as np


@six.add_metaclass(abc.ABCMeta)
class Metric:
    def __init__(self):
        self._corr = None

    @property
    def corr(self):
        return self._corr

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class AndersonDarling(Metric):
    def __init__(self):
        super(AndersonDarling, self).__init__()
        self._crit = None
        self._p = None

    def __call__(self, a, v, *args, **kwargs):
        self._corr, self._crit, self._p = stats.anderson_ksamp([a, v])

        return self.corr


class KendallsTau(Metric):
    def __init__(self):
        super(KendallsTau, self).__init__()
        self._p = None

    def __call__(self, a, v, *args, **kwargs):
        self._corr, self._p = stats.kendalltau(a, v, *args, **kwargs)

        return self.corr


class KolmogorovSmirnov(Metric):
    def __init__(self):
        super(KolmogorovSmirnov, self).__init__()
        self._p = None

    def __call__(self, a, v, *args, **kwargs):
        self._corr, self._p = stats.ks_2samp(a, v)

        return self.corr


class CorrMatrixCoeff(Metric):
    def __call__(self, a, v, *args, **kwargs):
        self._corr = np.corrcoef(a, v)[0, 1]

        return self.corr


class Epsilon(Metric):
    def __call__(self, a, v, *args, **kwargs):
        norm_a = a / np.ma.sqrt(np.ma.sum(a ** 2))
        norm_v = v / np.ma.sqrt(np.ma.sum(v ** 2))

        self._corr = np.ma.sum(norm_a * norm_v)

        return self.corr


class CrossCorrelate(Metric):
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
    def __call__(self, a, v, *args, **kwargs):
        # if normalize:
        a = (a - np.ma.mean(a)) / (np.ma.std(a) * a.size)
        v = (v - np.ma.mean(v)) / np.ma.std(v)

        self._corr = np.correlate(a, v)

        return self.corr