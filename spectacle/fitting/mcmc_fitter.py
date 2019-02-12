import logging

import numpy as np
import uuid
from astropy.modeling.models import Linear1D, Const1D
from astropy.modeling.fitting import LevMarLSQFitter, Fitter, _model_to_fit_params, _fitter_to_model_params
from astropy.modeling import Parameter
from scipy import stats
import scipy.optimize as op
import emcee

from ..utils.misc import find_nearest


class MCMCFitter:

    def lnprior(self, theta, model):
        # Convert the array of parameter values back into model parameters
        _, fit_params_indices = _model_to_fit_params(model)
        bounds = np.array(list(model.bounds.values()))[fit_params_indices]
        bounds = list(bounds)
        bounds.append((-10.0, 1.0))

        # Compose a list of all the `Parameter` objects, so we can still
        # reference their bounds information
        # params = [getattr(model, x)
        #           for x in np.array(model.param_names)[fit_params_indices]]

        if all([(bounds[i][0] or -np.inf)
                        <= theta[i] <= (bounds[i][1] or np.inf)
                for i in range(len(theta))]):
            return 0.0

        return -np.inf

    def lnlike(self, theta, x, y, yerr, model):
        # Convert the array of parameter values back into model parameters
        _fitter_to_model_params(model, theta[:-1])
        lnf = theta[-1]

        mod_y = model(x)
        inv_sigma2 = 1.0 / (yerr ** 2 + mod_y ** 2 * np.exp(2 * lnf))
        res = -0.5 * (np.nansum((y - mod_y) ** 2 * inv_sigma2 - np.log(inv_sigma2)))
        return res

    def lnprob(self, theta, x, y, yerr, model):
        model = model.copy()
        lp = self.lnprior(theta, model)

        if not np.isfinite(lp):
            return -np.inf

        ll = self.lnlike(theta, x, y, yerr, model)

        return lp + ll


class EmceeFitter(MCMCFitter):
    def __call__(self, model, x, y, yerr=None, nwalkers=500, steps=200):
        # If no errors are provided, assume all errors are normalized
        if yerr is None:
            yerr = np.zeros(shape=x.shape)

        # Retrieve the parameters that are not considered fixed or tied
        fit_params, fit_params_indices = _model_to_fit_params(model)
        fit_params = np.append(fit_params, np.log(0.1))
        # fit_params_indices = np.array(fit_params_indices).astype(int)

        # Perform a quick optimization of the parameters
        # nll = lambda *args: -self.lnlike(*args)
        #
        # result = op.minimize(nll, fit_params, args=(x, y, yerr, model))
        # fit_params = result["x"]
        # print(fit_params)

        # Cache the number of dimensions of the problem, and walker count
        ndim = len(fit_params)

        # Initialize starting positions of walkers in a Gaussian ball
        pos = [fit_params * (1 + 1e-1 * np.random.randn(ndim))
               for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob,
                                        args=(x, y, yerr, model), threads=8)
        sampler.run_mcmc(pos, steps, rstate0=np.random.get_state())

        # for result in sampler.sample(pos, iterations=steps, storechain=False):
        #     position = result[0]

        #     with open("chain.dat", "a") as f:
        #         for k in range(position.shape[0]):
        #             f.write("{0:4d} {1:s}\n".format(k, " ".join(position[k])))

        burnin = int(steps * 0.2)
        samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

        # Compute the quantiles.
        samples[:, -1] = np.exp(samples[:, -1])

        res = list(map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                       zip(*np.percentile(samples, [16, 50, 84], axis=0))))

        theta = [x[0] for x in res]

        _fitter_to_model_params(model, theta[:-1])
        print(theta)

        # fit_params, fit_params_indices = _model_to_fit_params(model)
        # model.parameters[fit_params_indices] = theta

        # for name, value in zip(np.array(model.param_names)[fit_params_indices], fit_params):
        #     print("{:20}: {:g}".format(name, value))

        return model


# class MultiNestFitter(MCMCFitter):
#     def __call__(self, model, x, y, yerr=None, nwalkers=100, steps=500):
#         # If no errors are provided, assume all errors are normalized
#         if yerr is None:
#             yerr = np.zeros(shape=x.shape)
#             # yerr.fill(1e-20)
#
#         self.__class__.model = model
#
#         # Retrieve the parameters that are not considered fixed or tied
#         fit_params, fit_params_indices = _model_to_fit_params(model)
#
#         # fit_params = np.append(fit_params, np.log(1e-20))
#         fit_params_indices = np.array(fit_params_indices).astype(int)
#
#         # Cache the number of dimensions of the problem, and walker count
#         ndim = len(fit_params)
#
#         from pymultinest.solve import solve
#         import os
#
#         if not os.path.exists("chains"):
#             os.mkdir("chains")
#
#         result = solve(LogLikelihood=self.lnlike, Prior=self.lnprior,
#                        n_dims=ndim, outputfiles_basename="chains/3-",
#                        verbose=True)
#
#         theta = [x.mean() for x in zip(fit_params, result['samples'].transpose())]
#
#         _fitter_to_model_params(model, theta)#[:-1])
#
#         # fit_params, fit_params_indices = _model_to_fit_params(model)
#         model.parameters[fit_params_indices] = theta
#
#         # for name, value in zip(np.array(model.param_names)[fit_params_indices], fit_params):
#         #     print("{:20}: {:g}".format(name, value))
#
#         return model