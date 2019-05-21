import emcee
import numpy as np
import scipy.optimize as op

from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.fitting import (_fitter_to_model_params,
                                      _model_to_fit_params)
from astropy.table import QTable
from pathos.multiprocessing import Pool


def lnprior(theta, model):
    # Convert the array of parameter values back into model parameters
    _, fit_params_indices = _model_to_fit_params(model)
    bounds = np.array(list(model.bounds.values()))[fit_params_indices]

    if all([(bounds[i][0] or -np.inf)
                    <= theta[i] <= (bounds[i][1] or np.inf)
            for i in range(len(theta))]):
        return 0.0

    return -np.inf


def lnlike(theta, x, y, yerr, model):
    # Convert the array of parameter values back into model parameters
    _fitter_to_model_params(model, theta)

    mod_y = model(x)

    # inv_sigma2 = 1.0 / (yerr ** 2 + mod_y ** 2 * np.exp(2 * lnf))
    # res = -0.5 * (np.sum((y - mod_y) ** 2 * inv_sigma2 - np.log(inv_sigma2)))
    if np.sum(yerr) > 0:
        res = -0.5 * np.sum(((y - mod_y)/yerr) ** 2)
    else:
        res = -0.5 * np.sum((y - mod_y) ** 2)

    return res


def lnprob(theta, x, y, yerr, model):
    lp = lnprior(theta, model)

    if not np.isfinite(lp):
        return -np.inf

    ll = lnlike(theta, x, y, yerr, model)

    return lp + ll


class EmceeFitter:
    def __init__(self):
        self._uncertainties = {}

    @property
    def uncertainties(self):
        return self._uncertainties

    @property
    def errors(self):
        tab = QTable([self._uncertainties['param_names'],
                      self._uncertainties['param_fit'],
                      self._uncertainties['param_err'][:, 0],
                      self._uncertainties['param_err'][:, 1],
                      self._uncertainties['param_units']],
                     names=('name', 'value', 'min_uncert', 'max_uncert', 'unit'))

        return tab

    def __call__(self, model, x, y, yerr=None, nwalkers=500, steps=200,
                 nprocs=1):
        model = model.copy()

        # If no errors are provided, assume all errors are normalized
        if yerr is None:
            yerr = np.zeros(shape=x.shape)

        # Retrieve the parameters that are not considered fixed or tied
        fit_params, fit_params_indices = _model_to_fit_params(model)
        # fit_params = np.append(fit_params, np.log(1))

        # TODO: The following two seem to be unnecessary, however, the fits
        # don't work well without *both* of them. Need to rethink.
        fitter = LevMarLSQFitter()
        fit_model = fitter(model, x, y)
        fit_params = fit_model.parameters[fit_params_indices]

        # Perform a quick optimization of the parameters
        nll = lambda *args: -lnlike(*args)
        result = op.minimize(nll, fit_params, args=(x, y, yerr, model))
        fit_params = result["x"]

        _fitter_to_model_params(model, fit_params)

        # Cache the number of dimensions of the problem, and walker count
        ndim = len(fit_params)

        # Initialize starting positions of walkers in a Gaussian ball
        pos = [fit_params * (1 + 1e-8 * np.random.randn(ndim))
               for _ in range(nwalkers)]

        with Pool(nprocs) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                            args=(x, y, yerr, model),
                                            pool=pool)
            sampler.run_mcmc(pos, steps)

        # for result in sampler.sample(pos, iterations=steps, storechain=False):
        #     position = result[0]

        #     with open("chain.dat", "a") as f:
        #         for k in range(position.shape[0]):
        #             f.write("{0:4d} {1:s}\n".format(k, " ".join(position[k])))

        burnin = int(steps * 0.1)
        samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

        # Compute the quantiles.
        res = list(map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                       zip(*np.percentile(samples, [16, 50, 84], axis=0))))

        theta = [x[0] for x in res]

        _fitter_to_model_params(model, theta)

        self._uncertainties['param_names'] = model.param_names
        self._uncertainties['param_fit'] = model.parameters

        errs = np.array([(0.0, 0.0) for _ in range(len(model.parameters))])
        errs[fit_params_indices] = np.array([(res[i][1], res[i][2])
                                    for i in range(len(res))])

        self._uncertainties['param_err'] =  errs
        self._uncertainties['param_units'] = [getattr(model, p).unit
                                              for p in model.param_names]

        return model
