import abc

import numpy as np
import six
from astropy.modeling import fitting, models
from astropy.modeling.fitting import LevMarLSQFitter, Fitter, SLSQPLSQFitter
from astropy.table import Table
from scipy import optimize, stats
import peakutils

from ..core.spectra import Spectrum1D
from ..modeling.models import Absorption1D, Voigt1D
from ..core.utils import find_nearest, find_bounds
from ..core.registries import line_registry

import logging


class LevMarFitter(LevMarLSQFitter):
    def __init__(self, *args, **kwargs):
        super(LevMarFitter, self).__init__(*args, **kwargs)

    @staticmethod
    def _find_local_edges(model, x, y):
        lind, rind = find_bounds(x, y, model.lambda_0)
        ledge, redge = x[lind], x[rind]

        return ledge, redge

    @classmethod
    def _find_local_peaks(cls, model, x, y):
        ledge, redge = cls._find_local_edges(model, x, y)
        mask = (x > ledge) & (x < redge)

        minds = peakutils.indexes(1 - y[mask])
        rinds = list(map(lambda i: find_nearest(y, y[mask][i]), minds))
        peaks = list(map(lambda i: x[i], rinds))

        return peaks, ledge, redge

    @classmethod
    def initialize(cls, model, x, y):
        """
        Attempt to analytically determine a best initial guess for each line
        in the model.
        """
        peaks, ledge, redge = cls._find_local_peaks(model, x, y)

        # Calculate the eq
        temp_spec = Spectrum1D(y, dispersion=x)
        ew, _ = temp_spec.equivalent_width(x_range=(ledge, redge))

        # Guess the column density
        # N = ew * 50 / model.f_value
        # print("N", N)

        # Guess doppler
        b_ratio = (redge - ledge) / len(peaks) / ew

        # Override the initial guesses on the models
        model.v_doppler = model.v_doppler * b_ratio
        model.lambda_0.bounds = (ledge, redge)

    def __call__(self, model, x, y, initialize=True, *args, **kwargs):
        if initialize:
            if hasattr(model, '_submodels'):
                for i, mod in enumerate(model):
                    if isinstance(mod, Voigt1D):
                        self.initialize(mod, x, y)
            else:
                if isinstance(model, Voigt1D):
                    self.initialize(model, x, y)

        return super(LevMarFitter, self).__call__(model, x, y, *args, **kwargs)


class DynamicLevMarFitter(LevMarFitter):
    def __init__(self, *args, **kwargs):
        super(DynamicLevMarFitter, self).__init__(*args, **kwargs)

        self._chisq = 100
        self._p_value = 0

    @property
    def chisq(self):
        return self._chisq

    @property
    def p_value(self):
        return self._p_value

    @staticmethod
    def chi2(mod, x, y):
        ddof = len(list(
            filter(lambda x: x == False,
                   mod.fixed.values())))

        mod_y = mod(x).data
        mod_y[mod_y==0] = 1e-10
        y[y==0] = 1e-10

        res = stats.chisquare(f_obs=mod(x).data,
                              f_exp=y,
                              ddof=ddof)

        return res

    def __call__(self, model, x, y, *args, **kwargs):
        fit_mod = super(DynamicLevMarFitter, self).__call__(
            model, x, y, *args, **kwargs)

        self._chisq, self._p_value = self.chi2(fit_mod, x, y)

        # First try reducing the number of lines
        while self._chisq > 0.1:
            fit_sub_mods = [fm.copy() for fm in fit_mod]
            fit_sub_mods = fit_sub_mods[:-1] if len(fit_sub_mods) > 1 \
                                             else fit_sub_mods

            mod = Absorption1D(lines=fit_sub_mods[1:],
                               continuum=fit_sub_mods[0])

            temp_fit_mod = super(DynamicLevMarFitter, self).__call__(
                mod, x, y, *args, **kwargs)

            chisq, p = self.chi2(temp_fit_mod, x, y)

            if chisq <= self._chisq and np.abs(chisq - self._chisq) >= 0.1:
                logging.info(
                    "Fit improved by removing line at {}:"
                    "\n\tChi squared: {} -> {}.".format(
                        fit_mod[-1].lambda_0.value + fit_mod[-1].delta_lambda.value,
                        self._chisq, chisq))

                self._chisq, self._p_value = chisq, p
                fit_mod = temp_fit_mod
            else:
                logging.info(
                    "Fit did not improve by removing line:"
                    "\n\tChi squared: {} -> {}.".format(
                        self._chisq, chisq))
                break
        else:
            logging.info("Fit result is below chi squared of 0.1 ({}).".format(
                self._chisq))

        # Then try adding lines
        while self._chisq > 0.1:
            diff_ind = np.argmax(np.abs(y - fit_mod(x).data))

            new_line = fit_mod[1].copy()
            new_line.delta_lambda = x[diff_ind] - fit_mod[1].lambda_0

            fit_sub_mods = [fm.copy() for fm in fit_mod] + [new_line]
            mod = Absorption1D(lines=fit_sub_mods[1:],
                               continuum=fit_sub_mods[0])

            temp_fit_mod = super(DynamicLevMarFitter, self).__call__(
                mod, x, y, *args, **kwargs)

            chisq, p = self.chi2(temp_fit_mod, x, y)

            if chisq <= self._chisq and np.abs(chisq - self._chisq) >= 0.1:
                logging.info(
                    "Fit improved with addition of line at {}:"
                    "\n\tChi squared: {} -> {}.".format(
                        new_line.lambda_0.value + new_line.delta_lambda.value,
                        self._chisq, chisq))

                self._chisq, self._p_value = chisq, p
                fit_mod = temp_fit_mod
            else:
                logging.info(
                    "Fit did not improve with additional line:"
                    "\n\tChi squared: {} -> {}.".format(
                        self._chisq, chisq))
                break
        else:
            logging.info("Fit result is below chi squared of 0.1 ({}).".format(
                self._chisq))

        return fit_mod
