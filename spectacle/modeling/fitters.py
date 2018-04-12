import logging

import numpy as np
import uuid
from astropy.modeling.fitting import LevMarLSQFitter, Fitter, _model_to_fit_params, _fitter_to_model_params
from astropy.modeling import Parameter
from scipy import stats
import emcee

import matplotlib.pyplot as plt

from ..utils import find_nearest


class MCMCFitter:
    @classmethod
    def lnprior(cls, theta, model):
        _fitter_to_model_params(model, theta[:-1])

        _, fit_params_indices = _model_to_fit_params(model)

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
    def lnlike(cls, theta, x, y, yerr, model, ax):
        # Convert the array of parameter values back into model parameters
        _fitter_to_model_params(model, theta[:-1])

        mod_y = model(x)

        ax.plot(x, mod_y, color='k', alpha=0.2)

        inv_sigma2 = 1.0 / (yerr ** 2 + mod_y ** 2 * np.exp(2 * theta[-1]))

        res = -0.5 * (
            np.nansum((y - mod_y) ** 2 * inv_sigma2 - np.log(inv_sigma2)))

        if np.isnan(res):
            print(model)
            print("f: ", theta[-1])
            print(np.sum(1/(yerr ** 2 + mod_y ** 2 * np.exp(2 * theta[-1]))))

            print(np.sum(inv_sigma2))
            print(np.sum(log_inv_sigma2))
            print(np.sum(mod_diff))

        return res

    @classmethod
    def lnprob(cls, theta, x, y, yerr, model, ax):
        model = model.copy()
        lp = cls.lnprior(theta, model)

        if not np.isfinite(lp):
            return -np.inf

        ll = cls.lnlike(theta, x, y, yerr, model, ax)

        return lp + ll

    def __call__(self, model, x, y, yerr=None, nwalkers=100, steps=500):
        # If no errors are provided, assume all errors are normalized
        if yerr is None:
            yerr = np.zeros(shape=x.shape)

        # Retrieve the parameters that are not considered fixed or tied
        for sm in model:
            if hasattr(sm, 'v_doppler'):
                print(sm.v_doppler, end='\n')

        fit_params, fit_params_indices = _model_to_fit_params(model)

        for name, value in zip(np.array(model.param_names)[fit_params_indices], fit_params):
            print("{:20}: {:g}".format(name, value))

        fit_params = np.append(fit_params, np.log(0.5))
        fit_params_indices = np.array(fit_params_indices).astype(int)

        # Cache the number of dimensions of the problem, and walker count
        ndim = len(fit_params)



        import matplotlib.pyplot as plt

        f, ax = plt.subplots()

        ax.plot(x, y)


        # Initialize starting positions of walkers in a Gaussian ball
        pos = [fit_params * (1 + 0.01 * np.random.randn(ndim))
               for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, MCMCFitter.lnprob,
                                        args=(x, y, yerr, model, ax))
        sampler.run_mcmc(pos, steps, rstate0=np.random.get_state())


        plt.show()

        print("Shape: ", sampler.chain.shape)

        import matplotlib.pyplot as plt

        f, (ax1, ax2, ax3) = plt.subplots(3, 1)

        for i in range(sampler.chain.shape[0]):
            ax1.plot(sampler.chain[i, :, 0])
            ax2.plot(sampler.chain[i, :, 1])
            ax3.plot(sampler.chain[i, :, 2])

        plt.tight_layout()
        plt.savefig("test.png")

        burnin = int(steps * 0.1)
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

        fit_params, fit_params_indices = _model_to_fit_params(model)

        for sm in model:
            if hasattr(sm, 'v_doppler'):
                print(sm.v_doppler, end='\n')

        for name, value in zip(np.array(model.param_names)[fit_params_indices], fit_params):
            print("{:20}: {:g}".format(name, value))

        return model
