import peakutils
from scipy.signal import savgol_filter
from astropy.modeling import fitting, models
import numpy as np
from .optimizer import optimize


class Modeler:
    def __init__(self, fit_method='LevMarLSQFitter', noise=3.5, min_dist=100):
        self.fit_method = fit_method
        self.detrend = False
        self.noise = 3.5
        self.min_dist = 100

    def __call__(self, spectrum, detrend=None):
        self.spectrum = spectrum.copy()

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
                               intercept=np.median(self.spectrum.flux))
        fitter = getattr(fitting, mode)()
        cont_fit = fitter(cont, self.spectrum.dispersion, self.spectrum.flux,
                          weights=1/(np.abs(np.median(self.spectrum.flux) -
                                            self.spectrum.flux)) ** 3)

        self.spectrum._continuum_model = cont_fit

    def find_lines(self):
        continuum = self.spectrum.continuum if self.spectrum.continuum is not None \
                                       else np.median(self.spectrum.flux)

        inv_flux = continuum - self.spectrum.flux
        print(np.std(inv_flux)*self.noise/np.max(inv_flux))
        indexes = peakutils.indexes(inv_flux,
                                    thres=np.std(inv_flux)*self.noise/np.max(
                                        inv_flux),
                                    min_dist=self.min_dist)
        indexes = np.array(indexes)

        print("Found {} peaks".format(len(indexes)))

        import matplotlib.pyplot as plt
        plt.plot(self.spectrum.dispersion, self.spectrum.flux)

        for cind in indexes:
            plt.plot(self.spectrum.dispersion[cind],
                     self.spectrum.flux[cind], marker='o')
            amp, mu, gamma = (inv_flux[cind],
                              self.spectrum.dispersion[cind],
                              1)

            self.spectrum.add_line(lambda_0=mu, f_value=0.5, gamma=1e12,
                                   v_doppler=1e7, column_density=1e14)
        plt.plot(self.spectrum.ideal_flux)

        plt.show()

        print("Finished applying lines")

    def detrend_spectrum(self):
        self.spectrum._flux = savgol_filter(self.spectrum.flux, 5, 2)

    def fit_models(self):
        if self.fit_method != 'mcmc':
            fitter = getattr(fitting, self.fit_method)()
            model = self.spectrum.model
            model_fit = fitter(model, self.spectrum.dispersion,
                               self.spectrum.flux, maxiter=2000)

            self.spectrum.model.parameters = model_fit.parameters
        else:
            optimize(self.spectrum)
