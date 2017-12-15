import logging

import numpy as np
import peakutils
from astropy.modeling.fitting import LevMarLSQFitter, Fitter, _model_to_fit_params, _fitter_to_model_params
from astropy.modeling import Parameter
from scipy import stats
import emcee

from ..utils import find_nearest


class MCMCFitter:
    @classmethod
    def lnprior(cls, theta, model):
        _fitter_to_model_params(model, theta[:-1])
        fit_params, fit_params_indices = _model_to_fit_params(model)

        # Compose a list of all the `Parameter` objects, so we can still
        # reference their bounds information
        params = [getattr(model, x)
                  for x in np.array(model.param_names)[
                      np.array(fit_params_indices)]]

        if all([(params[i].bounds[0] or -np.inf)
                        <= theta[i] <= (params[i].bounds[1] or np.inf)
                for i in range(len(theta[:-1]))]) and -10.0 < theta[-1] < 1.0:
            return 0.0

        return -np.inf

    @classmethod
    def lnlike(cls, theta, x, y, yerr, model):
        try:
            # Convert the array of parameter values back into model parameters
            _fitter_to_model_params(model, theta[:-1])

            mod_y = model(x, y)

            inv_sigma2 = 1.0 / (yerr ** 2 + mod_y ** 2 * np.exp(2 * theta[-1]))

            res = -0.5 * (
                np.sum((y - mod_y) ** 2 * inv_sigma2 - np.log(inv_sigma2)))

            return res
        except:
            logging.error("Something bad happened.")

            return 0

    @classmethod
    def lnprob(cls, theta, x, y, yerr, model):
        model = model.copy()

        lp = cls.lnprior(theta, model)

        if not np.isfinite(lp):
            return -np.inf

        return lp + cls.lnlike(theta, x, y, yerr, model)

    def __call__(self, model, x, y, yerr=None):
        # If no errors are provided, assume all errors are normalized
        if yerr is None:
            yerr = np.ones(shape=x.shape)

        # Retrieve the parameters that are not considered fixed or tied
        fit_params, fit_params_indices = _model_to_fit_params(model)
        fit_params = np.append(fit_params, 0.5)

        # Cache the number of dimensions of the problem, and walker count
        ndim, nwalkers = len(fit_params), 20

        # Initialize starting positions of walkers in a Gaussian ball
        pos = [fit_params + 1e-2 * np.random.randn(ndim)
               for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob,
                                        args=(x, y, yerr, model))

        sampler.run_mcmc(pos, 20, rstate0=np.random.get_state())

        burnin = 1
        samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

        # Compute the quantiles.
        samples[:, 2] = np.exp(samples[:, 2])
        res = map(
            lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
            zip(*np.percentile(samples, [16, 50, 84],
                               axis=0)))
        theta = [x[1] for x in res]

        model = model.copy()
        _fitter_to_model_params(model, theta[:-1])

        return model
