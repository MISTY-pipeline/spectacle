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
    def initialize(cls, model, x, y, find_peaks):
        """
        Attempt to analytically determine a best initial guess for each line
        in the model.
        """
        if find_peaks:
            peaks, ledge, redge = cls._find_local_peaks(model, x, y)
            lam_ind = find_nearest(peaks, model.lambda_0)
            flambda_0 = peaks[lam_ind]

            model.lambda_0 = flambda_0
        else:
            ledge, redge = cls._find_local_edges(model, x, y)

        # Calculate the eq
        temp_spec = Spectrum1D(y, dispersion=x)
        ew, _ = temp_spec.equivalent_width(x_range=(ledge, redge))

        # Guess the column density
        # N = ew * 10 / model.f_value
        # print("N", N)

        # Guess doppler
        b_ratio = (redge - ledge) / ew

        # Override the initial guesses on the models
        model.v_doppler = model.v_doppler * b_ratio
        model.lambda_0.bounds = (ledge, redge)
        # model.column_density = model.column_density * (2 - y[lam_ind])

    def __call__(self, model, x, y, initialize=True, find_peaks=True, *args,
                 **kwargs):
        if initialize:
            if hasattr(model, '_submodels'):
                for mod in model:
                    if isinstance(mod, Voigt1D):
                        self.initialize(mod, x, y, find_peaks=find_peaks)
            else:
                if isinstance(model, Voigt1D):
                    self.initialize(model, x, y, find_peaks=find_peaks)

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
    def red_chi2(mod, x, y):
        ddof = len(list(
            filter(lambda x: x == False,
                   mod.fixed.values())))

        res = stats.chisquare(f_obs=mod(x).data,
                              f_exp=y,
                              ddof=ddof)

        return res

    def __call__(self, model, x, y, *args, **kwargs):
        # Estimate the signal-to-noise of the spectrum
        ddof = len(list(
            filter(lambda p: p == False,
                   model.fixed.values())))

        sn = stats.signaltonoise(y, ddof=ddof)
        logging.info("Signal-to-noise ratio: {}".format(sn))

        # Compose spectrum object
        spectrum = Spectrum1D(y, dispersion=x)
        lines = spectrum.find_lines(threshold=1/sn, min_dist=10,
                                    interpolate=True)

        # Create model object, and fit it
        model = Absorption1D(lines=lines)

        fit_mod = super(DynamicLevMarFitter, self).__call__(
            model, x, y, find_peaks=False, *args, **kwargs)

        return fit_mod

    # def __call__(self, model, x, y, *args, **kwargs):
    #     def _compose_model(threshold=1.0, min_dist=10):
    #         spectrum = Spectrum1D(y, dispersion=x)
    #         lines = spectrum.find_lines(threshold=threshold,
    #                                     min_dist=min_dist)
    #         model = Absorption1D(lines=lines)
    #
    #         return model
    #
    #     threshold = 1.0
    #     fin_mod = _compose_model(threshold)
    #     self._chisq, self._p_value = self.red_chi2(fin_mod, x, y)
    #
    #     while self._chisq > 0.1 and threshold > 0.1:
    #         model = _compose_model(threshold)
    #
    #         fit_mod = super(DynamicLevMarFitter, self).__call__(
    #             model, x, y, *args, **kwargs)
    #
    #         chisq, p = self.red_chi2(fit_mod, x, y)
    #
    #         if chisq <= self._chisq:
    #             logging.info("Adding new line to improve fit. (threshold={}, "
    #                          "red_chisq={}/{}).".format(threshold, chisq,
    #                                                     self._chisq))
    #             fin_mod = fit_mod
    #             self._chisq, self._p_value = chisq, p
    #         else:
    #             logging.info("For threshold={}, chisq did not improve."
    #                          "(red_chi2={}/{}).".format(threshold, chisq,
    #                                                     self._chisq))
    #             break
    #         threshold -= 0.2
    #     else:
    #         logging.info("Reached sufficient chisq or submodel limit.")
    #
    #     return fin_mod


@six.add_metaclass(fitting._FitterMeta)
class LevMarCurveFitFitter(object):
    """
    Levenberg-Marquardt algorithm and least squares statistic based on
    SciPy's `curve_fit` in order to allow passing in of data errors.

    Attributes
    ----------
    covar : 2d array
        The estimated covariance of the optimal parameter values. The
        diagonals provide the variance of the parameter estimate. See
        `scipy.optimize.curve_fit` documentation for more details.

    """

    # The constraint types supported by this fitter type.
    supported_constraints = ['fixed', 'tied', 'bounds']

    def __init__(self):
        self.fit_info = {'nfev': None,
                         'fvec': None,
                         'fjac': None,
                         'ipvt': None,
                         'qtf': None,
                         'message': None,
                         'ierr': None,
                         'param_jac': None,
                         'param_cov': None}

        super(LevMarCurveFitFitter, self).__init__()

    def objective_wrapper(self, model):
        def objective_function(dispersion, *param_vals):
            """
            Function to minimize.

            Parameters
            ----------
            fps : list
                parameters returned by the fitter
            args : list
                [model, [weights], [input coordinates]]
            """
            fitting._fitter_to_model_params(model, param_vals)

            return np.ravel(model(dispersion))

        return objective_function

    def __call__(self, model, x, y, z=None, sigma=None,
                 maxiter=fitting.DEFAULT_MAXITER, acc=fitting.DEFAULT_ACC,
                 epsilon=fitting.DEFAULT_EPS, estimate_jacobian=False,
                 **kwargs):
        """
        Fit data to this model.

        Returns
        -------
        model_copy : `~astropy.modeling.FittableModel`
            a copy of the input model with parameters set by the fitter
        """
        model_copy = fitting._validate_model(model, self.supported_constraints)

        if model_copy.fit_deriv is None or estimate_jacobian:
            dfunc = None
        else:
            dfunc = self._wrap_deriv

        init_values, _ = fitting._model_to_fit_params(model_copy)

        try:
            fit_params, cov_x = optimize.curve_fit(
                self.objective_wrapper(model_copy), x, y,
                # method='trf',
                p0=init_values,
                sigma=sigma,
                epsfcn=epsilon,
                maxfev=maxiter,
                col_deriv=model_copy.col_fit_deriv,
                xtol=acc,
                # absolute_sigma=True,
                **kwargs)
        except RuntimeError as e:
            logging.error(e)
            fit_params, _ = fitting._model_to_fit_params(model_copy)
            cov_x = []

        fitting._fitter_to_model_params(model_copy, fit_params)

        cov_diag = []

        for i in range(len(fit_params)):
            try:
                cov_diag.append(np.absolute(cov_x[i][i]) ** 0.5)
            except:
                cov_diag.append(0.00)

        self.fit_info['cov_x'] = cov_x
        self.fit_info['param_cov'] = cov_diag

        return model_copy

    @staticmethod
    def _wrap_deriv(params, model, weights, x, y, z=None):
        """
        Wraps the method calculating the Jacobian of the function to account
        for model constraints.
        `scipy.optimize.leastsq` expects the function derivative to have the
        above signature (parlist, (argtuple)). In order to accommodate model
        constraints, instead of using p directly, we set the parameter list in
        this function.
        """

        if weights is None:
            weights = 1.0

        if any(model.fixed.values()) or any(model.tied.values()):

            if z is None:
                full_deriv = np.ravel(weights) * np.array(
                    model.fit_deriv(x, *model.parameters))
            else:
                full_deriv = (np.ravel(weights) * np.array(
                    model.fit_deriv(x, y, *model.parameters)).T).T

            pars = [getattr(model, name) for name in
                    model.param_names]
            fixed = [par.fixed for par in pars]
            tied = [par.tied for par in pars]
            tied = list(
                np.where([par.tied is not False for par in pars],
                         True, tied))
            fix_and_tie = np.logical_or(fixed, tied)
            ind = np.logical_not(fix_and_tie)

            if not model.col_fit_deriv:
                full_deriv = np.asarray(full_deriv).T
                residues = np.asarray(
                    full_deriv[np.nonzero(ind)]).T
            else:
                residues = full_deriv[np.nonzero(ind)]

            return [np.ravel(_) for _ in residues]
        else:
            if z is None:
                return [np.ravel(_) for _ in
                        np.ravel(weights) * np.array(
                            model.fit_deriv(x, *params))]
            else:
                return [np.ravel(_) for _ in (
                np.ravel(weights) * np.array(model.fit_deriv(x, y, *params)).T).T]
