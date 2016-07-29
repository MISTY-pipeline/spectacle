import numpy as np
import emcee
from astropy.modeling.models import Voigt1D, Linear1D
from spectacle.core.spectra import Spectrum1D
from astropy.io import fits

import matplotlib.pyplot as plt
import corner


def lnprior(theta, y):
    slope, intercept = theta[0:2]
    x_0 = theta[2:-1:4]

    amp_L = theta[3:-1:4]
    amp_L_mask = (min(y) * 1.1 <= amp_L) & (amp_L <= intercept)

    fwhm_L = theta[4:-1:4]
    fwhm_G = theta[5:-1:4]

    lnf = theta[-1]

    if -10.0 <= lnf <= 1.0 and np.all(amp_L[amp_L_mask]) and min(y) <= \
            intercept <= max(y):
        return 0.0
    return -np.inf


def lnlike(theta, model, x, y, yerr):
    model.parameters = theta[:-1]
    lnf = theta[-1]

    pmodel = model(x)

    inv_sigma2 = 1.0 / (yerr ** 2 + pmodel ** 2 * np.exp(2 * lnf))

    return -0.5 * np.sum((y - pmodel) ** 2 * inv_sigma2 - np.log(inv_sigma2))


def lnprob(theta, model, x, y, yerr):
    lp = lnprior(theta, y)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, model, x, y, yerr)


def optimize(spectrum):
    # Grab original data
    disp, flux = spectrum.dispersion, spectrum.flux

    # Generate parameter list
    params = np.array(list(spectrum.model.parameters) + [0.5])

    # Set up the sampler
    ndim, nwalkers = len(params), 100 if len(params) * 2 <= 100 else len(
        params) * 2
    pos = [params * (1 + 1e-3 * np.random.randn(ndim)) for i in range(
        nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(spectrum.model, disp, flux,
                                          np.ones(disp.shape)),
                                    threads=8)

    # Clear and run the production chain.
    print("Running MCMC...")

    f = open("chain.dat", "w")
    f.close()

    for result in sampler.sample(pos, iterations=500, storechain=True,
                                 rstate0=np.random.get_state()):
        position = result[0]

        with open("chain.dat", "a") as f:
            for k in range(position.shape[0]):
                f.write("{0:4d} {1:s}\n".format(k, " ".join(map(str, position[
                                                                    k]))))

    print("Done.")

    burnin = 50
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

    # Compute the quantiles.
    samples[:, -1] = np.exp(samples[:, -1])
    vals = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],
                                                    axis=0)))
    vals = np.array(list(vals))
    print(vals)

    plt.plot(disp, flux)
    plt.plot(disp, spectrum.model(disp))

    # Show the walker-generated spectra
    for args in samples[np.random.randint(len(samples), size=100)]:
        v = spectrum.model
        v.parameters = args[:-1]
        plt.plot(disp, v(disp), color="k", alpha=0.1)

    spectrum.model.parameters = vals[:, 0][:-1]

    plt.plot(disp, spectrum.model(disp), color='r')
    plt.show()

    # Show the walkers' path for single variable
    # plt.plot(sampler.chain[:, :, 3].T, color="k", alpha=0.4)
    # plt.show()

    # Plot a corner plot
    # fig = corner.corner(samples, labels=["$x_0$", "$amp$", "$L$", "$G$",
    #                                      "$\ln f$"],
    #                     truths=[5, -10, 0.25, 0.45, 0.5])
    # fig.savefig("triangle.png")