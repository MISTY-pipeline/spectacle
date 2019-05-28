import numpy as np
import scipy.optimize as opt
from astropy.modeling.fitting import (LevMarLSQFitter, _convert_input,
                                      _fitter_to_model_params,
                                      _model_to_fit_params, _validate_model,
                                      fitter_unit_support)
from astropy.modeling.optimizers import (DEFAULT_ACC, DEFAULT_EPS,
                                         DEFAULT_MAXITER)
from astropy.table import QTable

__all__ = ['CurveFitter']


class CurveFitter(LevMarLSQFitter):
    def __init__(self):
        super().__init__()

        self.fit_info.update({'param_err': None,
                              'param_fit': None,
                              'param_names': None,
                              'param_units': None})

    @property
    def uncertainties(self):
        tab = QTable([self.fit_info['param_names'],
                      self.fit_info['param_fit'],
                      self.fit_info['param_err'],
                      self.fit_info['param_units']],
                     names=('name', 'value', 'uncert', 'unit'))

        return tab

    def __call__(self, *args, method='curve', **kwargs):

        if method == 'curve':
            fit_model = self._curve_fit(*args, **kwargs)
        elif method == 'leastsq':
            fit_model = self._leastsq(*args, **kwargs)
        elif method == 'bootstrap':
            fit_model = self._bootstrap(*args, **kwargs)
        else:
            raise ValueError("No method named '{}'. Must be one of 'curve', "
                             "'leastsq', or 'bootstrap'.".format(method))


        self.fit_info['param_units'] = [getattr(fit_model, p).unit
                                        for p in fit_model.param_names]

        return fit_model

    @fitter_unit_support
    def _curve_fit(self, model, x, y, z=None, weights=None, yerr=None,
                   maxiter=DEFAULT_MAXITER, acc=DEFAULT_ACC,
                   epsilon=DEFAULT_EPS, estimate_jacobian=False):
        """
        Fit data to this model.

        Parameters
        ----------
        model : `~astropy.modeling.FittableModel`
            model to fit to x, y, z
        x : array
           input coordinates
        y : array
           input coordinates
        z : array (optional)
           input coordinates
        weights : array (optional)
            Weights for fitting.
            For data with Gaussian uncertainties, the weights should be
            1/sigma.
        maxiter : int
            maximum number of iterations
        acc : float
            Relative error desired in the approximate solution
        epsilon : float
            A suitable step length for the forward-difference
            approximation of the Jacobian (if model.fjac=None). If
            epsfcn is less than the machine precision, it is
            assumed that the relative errors in the functions are
            of the order of the machine precision.
        estimate_jacobian : bool
            If False (default) and if the model has a fit_deriv method,
            it will be used. Otherwise the Jacobian will be estimated.
            If True, the Jacobian will be estimated in any case.
        equivalencies : list or None, optional and keyword-only argument
            List of *additional* equivalencies that are should be applied in
            case x, y and/or z have units. Default is None.

        Returns
        -------
        model_copy : `~astropy.modeling.FittableModel`
            a copy of the input model with parameters set by the fitter
        """
        model_copy = _validate_model(model, self.supported_constraints)
        farg = (model_copy, weights, ) + _convert_input(x, y, z)

        if model_copy.fit_deriv is None or estimate_jacobian:
            dfunc = None
        else:
            dfunc = self._wrap_deriv

        init_values, finds = _model_to_fit_params(model_copy)

        def f(x, *p0, mod=model_copy):
            _fitter_to_model_params(mod, p0)
            return mod(x)

        fitparams, cov_x = opt.curve_fit(f, x, y, p0=init_values,
                                         sigma=yerr, epsfcn=epsilon, jac=dfunc,
                                         col_deriv=model_copy.col_fit_deriv,
                                         maxfev=maxiter, xtol=acc,
                                         absolute_sigma=False)

        error = []

        for i in range(len(fitparams)):
            try:
                error.append(np.absolute(cov_x[i][i]) ** 0.5)
            except:
                error.append(0.00)

        _fitter_to_model_params(model_copy, fitparams)
        _output_errors = np.zeros(model.parameters.shape)
        _output_errors[finds] = np.array(error)

        self.fit_info['cov_x'] = cov_x
        self.fit_info['param_names'] = model_copy.param_names
        self.fit_info['param_err'] = _output_errors
        self.fit_info['param_fit'] = model_copy.parameters

        # now try to compute the true covariance matrix
        if (len(y) > len(init_values)) and cov_x is not None:
            sum_sqrs = np.sum(self.objective_function(fitparams, *farg)**2)
            dof = len(y) - len(init_values)
            self.fit_info['param_cov'] = cov_x * sum_sqrs / dof
        else:
            self.fit_info['param_cov'] = None

        self.fit_info['param_units'] = [getattr(model_copy, p).unit
                                        for p in model_copy.param_names]

        return model_copy

    @fitter_unit_support
    def _leastsq(self, model, x, y, *args, **kwargs):
        model_copy = super().__call__(model, x, y, *args, **kwargs)
        init_values, _ = _model_to_fit_params(model)
        pfit, finds = _model_to_fit_params(model_copy)
        _output_errors = np.zeros(model.parameters.shape)
        pcov = self.fit_info['param_cov']

        error = []

        for i in range(len(pfit)):
            try:
                error.append(np.absolute(pcov[i][i]) ** 0.5)
            except:
                error.append(0.00)

        _output_errors[finds] = np.array(error)

        self.fit_info['param_names'] = model_copy.param_names
        self.fit_info['param_err'] = _output_errors
        self.fit_info['param_fit'] = model_copy.parameters

        return model_copy

    @fitter_unit_support
    def _bootstrap(self, model, x, y, z=None, yerr=0.0, weights=None, **kwargs):
        model_copy = super().__call__(model, x, y, **kwargs)

        init_values, _ = _model_to_fit_params(model)
        pfit, finds = _model_to_fit_params(model_copy)
        farg = (model_copy, weights,) + _convert_input(x, y, z)

        self._output_errors = np.zeros(model.parameters.shape)

        # Get the stdev of the residuals
        residuals = self.objective_function(pfit, *farg)
        sigma_res = np.std(residuals)

        sigma_err_total = np.sqrt(sigma_res ** 2 + yerr ** 2)

        # 100 random data sets are generated and fitted
        ps = []

        for i in range(10):
            rand_delta = np.random.normal(0., sigma_err_total, len(y))
            rand_y = y + rand_delta

            farg = (model_copy, weights,) + _convert_input(x, rand_y, z)

            rand_fit, rand_cov = opt.leastsq(
                self.objective_function,
                init_values,
                args=farg,
                full_output=False)

            ps.append(rand_fit)

        ps = np.array(ps)
        mean_pfit = np.mean(ps, 0)

        # You can choose the confidence interval that you want for your
        # parameter estimates:
        n_sigma = 1.  # 1sigma gets approximately the same as methods above
        # 1sigma corresponds to 68.3% confidence interval
        # 2sigma corresponds to 95.44% confidence interval
        err_pfit = n_sigma * np.std(ps, 0)

        _fitter_to_model_params(model_copy, mean_pfit)
        self._output_errors[finds] = np.array(err_pfit)

        self.fit_info['param_names'] = model_copy.param_names
        self.fit_info['param_err'] = self._output_errors
        self.fit_info['param_fit'] = model_copy.parameters
        self.fit_info['param_units'] = [getattr(model_copy, p).unit
                                        for p in model_copy.param_names]

        return model_copy