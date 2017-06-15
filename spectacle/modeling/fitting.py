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

            if chisq <= self._chisq:
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

            if chisq <= self._chisq:
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
