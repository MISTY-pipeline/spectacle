from spectacle.spectra import Spectrum1D
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from spectacle.analysis import Modeler
from astropy.modeling.models import Linear1D
from spectacle.models import Voigt1D


def compare():
    spectrum1 = Spectrum1D()
    spectrum1.add_line(x_0=1200, b=10, gamma=2e-7, f=0.5)
    # spectrum1.set_continuum("Linear1D", slope=0.0, intercept=1.0)

    for gamma in np.logspace(-4, -6):
        print(gamma)
        spectrum2 = Spectrum1D()
        spectrum2.add_line(x_0=1600, b=87, gamma=gamma, f=0.6)
        plt.plot(spectrum2.dispersion, spectrum2.flux)

    plt.plot(spectrum1.dispersion, spectrum1.flux)
    plt.show()


def fitting():
    hdulist = fits.open("/Users/nearl/projects/hst_proposal/QSOALS/3C066A/3C066A_coadd_FUVM_final_all.fits")
    disp, flux = hdulist[1].data['WAVE'], hdulist[1].data['FLUX']

    flux = flux[(disp > 1350) & (disp < 1400)]
    disp = disp[(disp > 1350) & (disp < 1400)]

    spectrum = Spectrum1D(disp, flux)

    modeler = Modeler()
    result_spectrum = modeler(spectrum)

    plt.plot(spectrum.dispersion, spectrum.flux)
    plt.plot(result_spectrum.dispersion, result_spectrum.ideal_flux)
    plt.show()


if __name__ == '__main__':
    fitting()
    # compare()
