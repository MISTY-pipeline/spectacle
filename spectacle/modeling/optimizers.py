import six
import abc

import numpy as np
import emcee
from astropy.modeling.fitting import (_validate_model,
                                      _fitter_to_model_params,
                                      _model_to_fit_params, Fitter,
                                      _convert_input)


@six.add_metaclass(abc.ABCMeta)
class OptimizerFitter:
    def __init__(self):
        pass

    def objective_function(self, params):
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class PosteriorFitter(OptimizerFitter):
    supported_constraints = ['bounds', 'eqcons', 'ineqcons', 'fixed', 'tied']

    @staticmethod
    def ln_likelihood(params, model, x, y, yerr):
        _fitter_to_model_params(model, params)

        print(".", end=" ")

        y_mod = model(x).data

        # inv_sigma2 = 1.0 / (yerr ** 2 + y_mod ** 2 * np.exp(2))
        # print("inv_sigma: ", inv_sigma2.sum(), min(y_mod))
        #
        # diff = (y - y_mod)
        # print("diff: ", diff.sum())

        inv_sigma2 = 1.0/(yerr**2 + y_mod**2*np.exp(2))
        return -0.5*(np.sum((y-y_mod)**2*inv_sigma2 - np.log(inv_sigma2)))

        # res = -0.5 * (np.sum((y - y_mod) ** 2 * inv_sigma2 - np.log(inv_sigma2)))
        #
        # if np.isnan(res):
        #     print(list(zip(model.param_names, model.parameters)))

        # return res

    @staticmethod
    def ln_prior(params, model):
        _, fit_inds = _model_to_fit_params(model)

        # Check bounds. Assuming the order in theta is the same as parameters.
        if all(map(lambda p1, p2: p1 >= p2[0] and p1 <= p2[1], params,
                   np.array(list(model.bounds.values()))[fit_inds])):
            return 0.0
        return -np.inf

    @staticmethod
    def objective_function(params, model, x, y, yerr):
        lp = PosteriorFitter.ln_prior(params, model)

        if not np.isfinite(lp):
            return -np.inf

        return PosteriorFitter.ln_likelihood(params, model, x, y, yerr)

    def __call__(self, model, x, y, yerr=None, *args, **kwargs):
        model_copy = _validate_model(model, self.supported_constraints)
        init_values, _ = _model_to_fit_params(model_copy)

        if yerr is None:
            yerr = np.ones(x.shape)

        ndim, nwalkers = len(init_values), 100

        pos = [init_values * (1 + 1e-4 * np.random.randn(ndim)) for i in
               range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                        PosteriorFitter.objective_function,
                                        args=(model_copy, x, y, yerr))

        # Burn-in
        pos, prob, state = sampler.run_mcmc(pos, 200, rstate0=np.random.get_state())

        # reset the sampler
        sampler.reset()

        # Run sampler
        sampler.run_mcmc(pos, 500, rstate0=state)

        samples = sampler.chain[:, :, :].reshape((-1, ndim))

        # Compute the quantiles.
        # samples[:, 2] = np.exp(samples[:, 2])
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
        axes[0].plot(sampler.chain[:, :, 2].T, color="k", alpha=0.4)
        axes[0].set_ylabel("$m$")
        plt.savefig("para_output.png")


        fit_params = list(map(
            lambda v: v[0],
            zip(*np.percentile(samples, [16, 50, 84],
                               axis=0))))

        print(fit_params)

        _fitter_to_model_params(model_copy, fit_params)

        return model_copy
