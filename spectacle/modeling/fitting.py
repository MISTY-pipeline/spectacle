import peakutils
from scipy.signal import savgol_filter
from astropy.modeling import fitting, models
from astropy.extern import six
import numpy as np
from ..core.spectra import Spectrum1D

from .optimizer import optimize


class Fitter:
    def __init__(self, fit_method='LevMarLSQFitter', noise=3.5, min_dist=100):
        self.fit_method = fit_method
        self.detrend = False
        self.noise = 3.5
        self.min_dist = 100

    def __call__(self, spectrum, detrend=None):
        self.raw_spectrum = spectrum
        self.spectrum = Spectrum1D()

        if detrend is not None:
            self.detrend = detrend

        if self.detrend:
            self.detrend_spectrum()

        self.find_continuum()
        self.find_lines()
        self.fit_models()

        return self.spectrum

    def find_continuum(self, mode='LinearLSQFitter'):
        cont = models.Linear1D(slope=0.0,
                               intercept=np.median(self.raw_spectrum.flux))
        fitter = getattr(fitting, mode)()
        cont_fit = fitter(cont, self.raw_spectrum.dispersion,
                          self.raw_spectrum.flux,
                          weights=1/(np.abs(np.median(self.spectrum.flux) -
                                            self.spectrum.flux)) ** 3)

        self.spectrum._continuum_model = cont_fit

    def find_lines(self):
        continuum = self.spectrum.continuum if self.spectrum.continuum is not None \
                                       else np.median(self.raw_spectrum.flux)

        inv_flux = continuum - self.raw_spectrum.flux

        indexes = peakutils.indexes(inv_flux,
                                    thres=np.std(inv_flux)*self.noise/np.max(
                                        inv_flux),
                                    min_dist=self.min_dist)
        indexes = np.array(indexes)

        print("Found {} peaks".format(len(indexes)))

        import matplotlib.pyplot as plt
        plt.plot(self.raw_spectrum.dispersion, self.raw_spectrum.flux)

        for cind in indexes:
            plt.plot(self.raw_spectrum.dispersion[cind],
                     self.raw_spectrum.flux[cind], marker='o')
            amp, mu, gamma = (inv_flux[cind],
                              self.raw_spectrum.dispersion[cind],
                              1)

            self.spectrum.add_line(lambda_0=mu, f_value=0.5, gamma=1e12,
                                   v_doppler=1e7, column_density=1e14)
            # break
        plt.plot(self.spectrum.dispersion, self.spectrum.ideal_flux)

        plt.show()

        print("Finished applying lines")

    def detrend_spectrum(self):
        self.raw_spectrum._flux = savgol_filter(self.raw_spectrum.flux, 5, 2)

    def fit_models(self):
        if self.fit_method != 'mcmc':
            fitter = getattr(fitting, self.fit_method)()
            model = self.spectrum.model
            model_fit = fitter(model, self.raw_spectrum.dispersion,
                               self.raw_spectrum.flux, maxiter=2000)

            self.spectrum.model.parameters = model_fit.parameters

            pcov = fitter.fit_info['cov_x']

            if (len(self.raw_spectrum.flux) > len(model_fit.parameters)) and \
                            pcov is not None:
                s_sq = ((model_fit(self.raw_spectrum.dispersion) -
                         self.raw_spectrum.flux) ** 2).sum() / (
                len(self.raw_spectrum.flux) - len(model_fit.parameters))
                pcov *= s_sq
            else:
                pcov = np.inf

            error = []
            for i in range(len(model_fit.parameters)):
                try:
                    error.append(np.absolute(pcov[i][i]) ** 0.5)
                except:
                    error.append(0.00)

            print(list(zip(model_fit.param_names, model_fit.parameters,
                           error)))

        else:
            optimize(self.spectrum)


# @six.add_metaclass(fitting._FitterMeta)
# class LevMarCurveFitFitter(object):
#     """
#     Levenberg-Marquardt algorithm and least squares statistic based on
#     SciPy's `curve_fit` in order to allow passing in of data errors.
#
#     Attributes
#     ----------
#     covar : 2d array
#         The estimated covariance of the optimal parameter values. The
#         diagonals provide the variance of the parameter estimate. See
#         `scipy.optimize.curve_fit` documentation for more details.
#
#     """
#
#     # The constraint types supported by this fitter type.
#     supported_constraints = ['fixed', 'tied', 'bounds']
#
#     def __init__(self):
#         super(LevMarCurveFitFitter, self).__init__()
#
#     def objective_function(self, p0, x, y, func, yerr=0.2, **kwargs):
#         pfit, pcov = optimize.curve_fit(func, x, y, p0=p0, sigma=yerr,
#                                         epsfcn=0.0001, **kwargs)
#         error = []
#
#         for i in range(len(pfit)):
#             try:
#                 error.append(np.absolute(pcov[i][i])**0.5)
#             except:
#                 error.append(0.00)
#
#         pfit_curvefit = pfit
#         perr_curvefit = np.array(error)
#
#         return pfit_curvefit, perr_curvefit
#
#     def __call__(self, model, x, y, z=None, weights=None,
#                  maxiter=fitting.DEFAULT_MAXITER, acc=fitting.DEFAULT_ACC,
#                  epsilon=fitting.DEFAULT_EPS, estimate_jacobian=False):
#         """
#         Fit data to this model.
#         Parameters
#         ----------
#         model : `~astropy.modeling.FittableModel`
#             model to fit to x, y, z
#         x : array
#            input coordinates
#         y : array
#            input coordinates
#         z : array (optional)
#            input coordinates
#         weights : array (optional)
#            weights
#         maxiter : int
#             maximum number of iterations
#         acc : float
#             Relative error desired in the approximate solution
#         epsilon : float
#             A suitable step length for the forward-difference
#             approximation of the Jacobian (if model.fjac=None). If
#             epsfcn is less than the machine precision, it is
#             assumed that the relative errors in the functions are
#             of the order of the machine precision.
#         estimate_jacobian : bool
#             If False (default) and if the model has a fit_deriv method,
#             it will be used. Otherwise the Jacobian will be estimated.
#             If True, the Jacobian will be estimated in any case.
#         Returns
#         -------
#         model_copy : `~astropy.modeling.FittableModel`
#             a copy of the input model with parameters set by the fitter
#         """
#         from scipy import optimize
#
#         model_copy = fitting._validate_model(model,
#                                              self.supported_constraints)
#         farg = (model_copy, weights,) + fitting._convert_input(x, y, z)
#
#         if model_copy.fit_deriv is None or estimate_jacobian:
#             dfunc = None
#         else:
#             dfunc = self._wrap_deriv
#
#         init_values, _ = fitting._model_to_fit_params(model_copy)
#
#         fitparams, cov_x = optimize.curve_fit(
#             self.objective_function, init_values, args=farg,
#             Dfun=dfunc,
#             col_deriv=model_copy.col_fit_deriv, maxfev=maxiter,
#             epsfcn=epsilon,
#             xtol=acc, full_output=True)
#
#         pfit, pcov = optimize.curve_fit(, x, y, p0=p0, sigma=yerr,
#                                         epsfcn=0.0001, **kwargs)
#         error = []
#
#         for i in range(len(pfit)):
#             try:
#                 error.append(np.absolute(pcov[i][i]) ** 0.5)
#             except:
#                 error.append(0.00)
#
#         pfit_curvefit = pfit
#         perr_curvefit = np.array(error)
#
#         fitting._fitter_to_model_params(model_copy, fitparams)
#         self.fit_info['cov_x'] = cov_x
#
#         # now try to compute the true covariance matrix
#         # if (len(y) > len(init_values)) and cov_x is not None:
#         #     sum_sqrs = np.sum(
#         #         self.objective_function(fitparams, *farg) ** 2)
#         #     dof = len(y) - len(init_values)
#         #     self.fit_info['param_cov'] = cov_x * sum_sqrs / dof
#         # else:
#         #     self.fit_info['param_cov'] = None
#
#         return model_copy
#
#     @staticmethod
#     def _wrap_deriv(params, model, weights, x, y, z=None):
#         """
#         Wraps the method calculating the Jacobian of the function to account
#         for model constraints.
#         `scipy.optimize.leastsq` expects the function derivative to have the
#         above signature (parlist, (argtuple)). In order to accommodate model
#         constraints, instead of using p directly, we set the parameter list in
#         this function.
#         """
#
#         if weights is None:
#             weights = 1.0
#
#         if any(model.fixed.values()) or any(model.tied.values()):
#
#             if z is None:
#                 full_deriv = np.ravel(weights) * np.array(
#                     model.fit_deriv(x, *model.parameters))
#             else:
#                 full_deriv = (np.ravel(weights) * np.array(
#                     model.fit_deriv(x, y, *model.parameters)).T).T
#
#             pars = [getattr(model, name) for name in
#                     model.param_names]
#             fixed = [par.fixed for par in pars]
#             tied = [par.tied for par in pars]
#             tied = list(
#                 np.where([par.tied is not False for par in pars],
#                          True, tied))
#             fix_and_tie = np.logical_or(fixed, tied)
#             ind = np.logical_not(fix_and_tie)
#
#             if not model.col_fit_deriv:
#                 full_deriv = np.asarray(full_deriv).T
#                 residues = np.asarray(
#                     full_deriv[np.nonzero(ind)]).T
#             else:
#                 residues = full_deriv[np.nonzero(ind)]
#
#             return [np.ravel(_) for _ in residues]
#         else:
#             if z is None:
#                 return [np.ravel(_) for _ in
#                         np.ravel(weights) * np.array(
#                             model.fit_deriv(x, *params))]
#             else:
#                 return [np.ravel(_) for _ in (
#                 np.ravel(weights) * np.array(model.fit_deriv(x, y, *params)).T).T]
