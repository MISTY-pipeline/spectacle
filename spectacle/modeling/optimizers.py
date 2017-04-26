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

    def ln_likelihood(self, params, model, x, y, yerr):
        _fitter_to_model_params(model, params)
        y_mod = model(x).data

        inv_sigma2 = 1.0 / (yerr ** 2 + y_mod ** 2 * np.exp(2))
        # print("inv_sigma: ", inv_sigma2.sum(), min(y_mod))
        #
        # diff = (y - y_mod)
        # print("diff: ", diff.sum())

        res = -0.5 * (np.sum(
            (y - y_mod) ** 2 * inv_sigma2 - np.log(inv_sigma2)))

        if np.isnan(res):
            print(list(zip(model.param_names, model.parameters)))

        return res

    def ln_prior(self, params, model):
        _, fit_inds = _model_to_fit_params(model)

        # Check bounds. Assuming the order in theta is the same as parameters.
        if all(map(lambda p1, p2: p1 >= p2[0] and p1 <= p2[1], params,
                   np.array(list(model.bounds.values()))[fit_inds])):
            return 0.0
        return -np.inf

    def objective_function(self, params, model, x, y, yerr):
        lp = self.ln_prior(params, model)

        if not np.isfinite(lp):
            return -np.inf

        return lp + self.ln_likelihood(params, model, x, y, yerr)

    def __call__(self, model, x, y, yerr=None, *args, **kwargs):
        model_copy = _validate_model(model, self.supported_constraints)
        init_values, _ = _model_to_fit_params(model_copy)

        if yerr is None:
            yerr = np.ones(x.shape)

        ndim, nwalkers = len(init_values), 100

        pos = [init_values + 1e-4 * np.random.randn(ndim) for i in
               range(nwalkers)]

        self.sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                             self.objective_function,
                                             args=(model_copy, x, y, yerr))

        # Burn-in
        pos, prob, state = self.sampler.run_mcmc(pos, 200)

        # reset the sampler
        self.sampler.reset()

        # Run sampler
        self.sampler.run_mcmc(pos, 500, rstate0=state)

        samples = self.sampler.chain[:, :, :].reshape((-1, ndim))

        # Compute the quantiles.
        samples[:, 2] = np.exp(samples[:, 2])

        fit_params = map(
            lambda v: v,
            zip(*np.percentile(samples, [16, 50, 84],
                               axis=0)))

        _fitter_to_model_params(model_copy, fit_params)

        return model_copy