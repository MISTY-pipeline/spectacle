from spectacle.spectra import Spectrum1D
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from spectacle.analysis import Modeler
from astropy.modeling.models import Linear1D, Voigt1D


def compare():
    disp1 = np.arange(0, 10, 0.1)
    mod1 = Linear1D(intercept=1.0, slope=0.0) + Voigt1D(5, -10, 0.5, 0.25)
    flux1 = mod1(disp1)

    spectrum1 = Spectrum1D(disp1, flux1)

    disp2 = np.arange(0, 10, 0.1)
    mod2 = Linear1D(intercept=1.0, slope=0.0) + Voigt1D(2, -7.5, 0.25, 0.45)
    flux2 = mod2(disp2)

    spectrum2 = Spectrum1D(disp2, flux2)

    plt.step(spectrum1.dispersion, spectrum1.flux)
    plt.step(spectrum2.dispersion, spectrum2.flux)
    plt.show()


def fitting():
    hdulist = fits.open("/Users/nearl/projects/hst_proposal/QSOALS/3C066A/3C066A_coadd_FUVM_final_all.fits")
    disp, flux = hdulist[1].data['WAVE'], hdulist[1].data['FLUX']

    flux = flux[(disp > 1350) & (disp < 1400)]
    disp = disp[(disp > 1350) & (disp < 1400)]

    spectrum = Spectrum1D(disp, flux)

    modeler = Modeler(detrend=True)
    result_spectrum = modeler(spectrum)

    plt.plot(spectrum.dispersion, spectrum.flux)
    plt.plot(result_spectrum.dispersion, result_spectrum.ideal_flux)
    plt.show()


if __name__ == '__main__':
    # fitting()
    compare()