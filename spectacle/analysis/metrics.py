import abc
import six

from scipy import stats
import numpy as np

from ..core.region_finder import find_regions
from ..utils import find_nearest


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
        self._corr, self._crit, self._p = stats.anderson_ksamp([a, v], *args,
                                                               **kwargs)

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
        self._corr = np.sum((a * v) /
                            (np.sqrt((a ** 2).sum()) *
                             np.sqrt((v ** 2).sum())))

        return self.corr


class DeltaV90(Metric):
    """
    Calculates the velocity width of 90 percent of the apparent optical depth
    in an absorption region.
    """
    def __call__(self, x, y1, y2, exact=False):
        comp_widths = []

        for y in [y1, y2]:
            # reg = find_regions(y, continuum=np.zeros(y.shape))
            #
            # reg_widths = []
            #
            # for lr, rr in reg:
            #     a = y[lr:rr]
            #     mid = (a[-1] - a[0]) * 0.5

            if exact:
                x5 = np.interp(np.percentile(y, 5), sorted(y), x)
                x95 = np.interp(np.percentile(y, 95), sorted(y), x)
            else:
                x5 = x[find_nearest(sorted(y), np.percentile(y, 5))]
                x95 = x[find_nearest(sorted(y), np.percentile(y, 95))]
            print(x5, x95)
                # reg_widths.append((mid, x5, x95))

            comp_widths.append(x95 - x5)
        print(comp_widths)
        return comp_widths[0]/comp_widths[1]



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
        a = (a - np.mean(a)) / (np.std(a) * a.size)
        v = (v - np.mean(v)) / np.std(v)

        self._corr = np.correlate(a, v)

        return self.corr