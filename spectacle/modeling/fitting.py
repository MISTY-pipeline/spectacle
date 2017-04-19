import abc

import numpy as np
import six
from astropy.modeling import fitting, models
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.table import Table
from scipy import optimize, stats

from ..modeling.models import Absorption1D
from ..core.utils import find_nearest
from ..core.registries import line_registry

import logging


@six.add_metaclass(abc.ABCMeta)
class Fitter1D:
    """
    Base fitter class for routines that run on the
    :class:`spectacle.core.spectra.Spectrum1D` object.
    """
    def __init__(self):
        """
        The fitter class is responsible for finding absorption features using
        a basic peak finding utility. This is done only if the fitting routine
        is not supplied with a pre-made model object which contains user-
        supplied lines.

        Parameters
        ----------
        threshold : float
            Normalized threshold. Only the peaks with amplitude higher than the
            threshold will be detected.
        min_dist :
            The minimum distance between detected peaks.
        """
        self.fit_info = None
        self._model = None
        self._indices = []

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def find_continuum(spectrum, mask=None, mode='LinearLSQFitter'):
        if mask is None:
            mask = np.ones(spectrum.data.shape, dtype=bool)

        data_med = np.median(spectrum.data[mask])
        cont = models.Linear1D(slope=0.0, intercept=data_med)

        fitter = getattr(fitting, mode)()

        # We only want to use weights if there is an appreciable difference
        # between the median and the continuum
        diff = np.abs(data_med - spectrum.data[mask])
        weights = diff ** -3 if np.sum(diff) > np.min(spectrum.data[mask]) \
                             else None

        cont_fit = fitter(cont, spectrum.dispersion[mask], spectrum.data[mask],
                          weights=weights)

        cont = cont_fit(spectrum.dispersion)

        return cont

    @property
    def model(self):
        return self._model


class DynamicLevMarFitter(LevMarLSQFitter):
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

    def __call__(self,  model, x, y, *args, **kwargs):
        fit_mod = super(DynamicLevMarFitter, self).__call__(model, x, y, *args,
                                                            **kwargs)
        ddof = len(list(
            filter(lambda x: x == False,
                   fit_mod.fixed.values())))

        res = stats.chisquare(f_obs=fit_mod(x).data,
                              f_exp=y,
                              ddof=ddof)

        self._chisq, self._p_value = res

        # Perform a chi2 test, only when the model and the data are within a
        # certain confidence interval do we accept the number of lines. Other-
        # wise, we will keep adding lines, to a limit.
        while self._chisq > 0.1 and \
                        len(fit_mod._submodels) < len(model._submodels) + 10:
            new_line = model[-1].copy()
            # Jitter lambda
            new_line.lambda_0 += np.random.sample() * 0.1

            fit_line_list = [x.copy() for x in model]

            new_mod = Absorption1D(lines=fit_line_list[1:] + [new_line],
                                   continuum=model[0].copy())

            fit_new_mod = super(DynamicLevMarFitter, self).__call__(new_mod,
                                                                    x, y,
                                                                    *args,
                                                                    **kwargs)

            res = stats.chisquare(f_obs=fit_new_mod(x).data,
                                  f_exp=y,
                                  ddof=ddof)

            chisq, p = res

            if chisq < self._chisq:
                logging.info("Adding new line to improve fit.")
                fit_mod = fit_new_mod
                self._chisq, self._p_value = res
            else:
                logging.info("Chisq did not improve, halting iteration.")
                break
        else:
            logging.info("Reached sufficient chisq or submodel limit.")

        return fit_mod


class LevMarFitter(Fitter1D):
    def __call__(self, model, x, y, err=None):
        # Create fitter instance
        # fitter = LevMarCurveFitFitter()
        fitter = LevMarLSQFitter()

        model_fit = fitter(model,
                           x, #spectrum.dispersion,
                           y, #spectrum.data,
                           err, #sigma=spectrum.uncertainty,
                           maxiter=min(2000, 100 * len(model.model.submodel_names))
                           )

        # Grab the covariance matrix
        param_cov = fitter.fit_info['param_cov']

        if param_cov is None:
            param_cov = np.zeros(len(model_fit.param_names))

        # This is not robust. The covariance matrix does not include
        # constrained parameters, so we must insert some value for the
        # variance to make sure the number of parameters match.
        for i, val in enumerate(model_fit.param_names):
            if model_fit.tied.get(val) or model_fit.fixed.get(val):
                param_cov = np.insert(param_cov, i, 0)

        # Update fit info
        self.fit_info = Table([model_fit.param_names,
                               model.model.parameters,
                               model_fit.parameters,
                               param_cov],
                              names=("Parameter", "Original Value",
                                     "Fitted Value", "Uncertainty"))

        # Create new model instance with fitted params
        new_model = model.copy(from_model=model_fit)

        # self._model = model_fit

        return new_model, model_fit


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
