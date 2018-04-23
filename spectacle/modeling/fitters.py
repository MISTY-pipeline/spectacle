import logging

import numpy as np
import uuid
from astropy.modeling.models import Linear1D, Const1D
from astropy.modeling.fitting import LevMarLSQFitter, Fitter, _model_to_fit_params, _fitter_to_model_params
from astropy.modeling import Parameter
from scipy import stats
import scipy.optimize as op
import emcee

import matplotlib.pyplot as plt

from ..utils import find_nearest


class MCMCFitter:
    model = None

    @classmethod
    def lnprior(cls, theta):
        model = cls.model.__class__()

        # Convert the array of parameter values back into model parameters
        _, fit_params_indices = _model_to_fit_params(model)

        # Compose a list of all the `Parameter` objects, so we can still
        # reference their bounds information
        params = [getattr(model, x)
                  for x in np.array(model.param_names)[fit_params_indices]]

        if all([(params[i].bounds[0] or -np.inf)
                        <= theta[i] <= (params[i].bounds[1] or np.inf)
                for i in range(len(theta))]):
            return 0.0

        return -np.inf

    @classmethod
    def lnlike(cls, theta, x, y, yerr):
        model = cls.model.__class__()

        # Convert the array of parameter values back into model parameters
        # _fitter_to_model_params(model, theta)#[:-1])
        _, fit_params_indices = _model_to_fit_params(model)
        model.parameters[fit_params_indices] = theta

        mod_y = model(x)
        inv_sigma2 = 1.0 / (yerr ** 2 + mod_y ** 2)
        res = -0.5 * (np.nansum((y - mod_y) ** 2 * inv_sigma2 - np.log(inv_sigma2)))
        return res

    @classmethod
    def lnprob(cls, theta, x, y, yerr):
        lp = cls.lnprior(theta)

        if not np.isfinite(lp):
            return -np.inf

        ll = cls.lnlike(theta, x, y, yerr)

        return lp + ll

    def __call__(self, model, x, y, yerr=None, nwalkers=100, steps=500):
        # If no errors are provided, assume all errors are normalized
        if yerr is None:
            yerr = np.zeros(shape=x.shape)
            # yerr.fill(1e-20)

        self.__class__.model = model

        # Retrieve the parameters that are not considered fixed or tied
        fit_params, fit_params_indices = _model_to_fit_params(model)

        # fit_params = np.append(fit_params, np.log(1e-20))
        fit_params_indices = np.array(fit_params_indices).astype(int)

        # Cache the number of dimensions of the problem, and walker count
        ndim = len(fit_params)

        # Initialize starting positions of walkers in a Gaussian ball
        pos = [fit_params + fit_params * 0.01 * np.random.randn(ndim)
               for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, MCMCFitter.lnprob,
                                        args=(x, y, yerr), threads=8)
        sampler.run_mcmc(pos, steps, rstate0=np.random.get_state())

        # for result in sampler.sample(pos, iterations=steps, storechain=False):
        #     position = result[0]

        #     with open("chain.dat", "a") as f:
        #         for k in range(position.shape[0]):
        #             f.write("{0:4d} {1:s}\n".format(k, " ".join(position[k])))

        # Plot the samples
        import matplotlib.pyplot as plt

        f, axes = plt.subplots(ndim, 1)

        for i, ax in enumerate(axes):
            ax.plot(sampler.chain[:, :, i].T, color='k', alpha=0.25)

        plt.tight_layout(h_pad=0.0)
        # plt.savefig("test.png")

        burnin = int(steps * 0.1)
        samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

        # Compute the quantiles.
        # samples[:, -1] = np.exp(samples[:, -1])

        res = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                  zip(*np.percentile(samples, [16, 50, 84], axis=0)))

        theta = [x[0] for x in res]

        _fitter_to_model_params(model, theta)#[:-1])

        # fit_params, fit_params_indices = _model_to_fit_params(model)
        model.parameters[fit_params_indices] = theta

        # for name, value in zip(np.array(model.param_names)[fit_params_indices], fit_params):
        #     print("{:20}: {:g}".format(name, value))

        return model
