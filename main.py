from spectacle.spectra import Spectrum1D
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from spectacle.analysis import Modeler
from astropy.modeling.models import Linear1D
from spectacle.models import Voigt1D


def compare():
    spectrum1 = Spectrum1D()
    spectrum1.add_line(lambda_0=1.03192700E+03, f_value=0.4164, gamma=1e6,
                       v_doppler=1e7, column_density=10**14.66)
    # spectrum1.set_continuum("Linear1D", slope=0.0, intercept=1.0)

    # for gamma in np.logspace(12, 14, 10):
    #     spectrum2 = Spectrum1D()
    #     spectrum2.add_line(lambda_0=1.03192700E+03, f_value=0.4164, gamma=gamma,
    #                        v_doppler=1e7, column_density=5.25e13)
    #     plt.plot(spectrum2.dispersion, spectrum2.flux, 'b-')

    # for n in np.logspace(12, 14, 10):
    #     spectrum2 = Spectrum1D()
    #     spectrum2.add_line(lambda_0=1.03192700E+03, f_value=0.4164,
    #                        gamma=1e6,
    #                        v_doppler=1e7, column_density=n)
    #     plt.plot(spectrum2.dispersion, spectrum2.flux, 'b-')

    #
    # for f in np.linspace(0.1, 1.0, 10):
    #     spectrum2 = Spectrum1D()
    #     spectrum2.add_line(lambda_0=2.03192700E+03, f_value=f, gamma=1e6,
    #                        v_doppler=1e7, column_density=5.25e13)
    #     plt.plot(spectrum2.dispersion, spectrum2.flux, 'g-')

    f, (ax1, ax2) = plt.subplots(2, 1)

    log_n, log_w = [], []
    for n in np.logspace(7, 25, 20):
        spectrum2 = Spectrum1D()
        spectrum2.add_line(lambda_0=1.03192700E+03, f_value=0.4164, gamma=1e5,
                           v_doppler=1e6, column_density=n)
        print(np.log10(n), np.log10(spectrum2.equivalent_width(
            1.03192700E+03)))
        log_n.append(np.log10(n * 0.4164 * (1.03192700E+03 / 5000.0)))
        log_w.append(np.log10(spectrum2.equivalent_width(1.03192700E+03) /
                              1.03192700E+03))
        ax1.plot(spectrum2.dispersion, spectrum2.flux, 'g-')

    ax2.plot(log_n, log_w)
    ax2.set_xlabel("$\log N f \lambda$")
    ax2.set_ylabel("$\log W / \lambda$")

    # plt.plot(spectrum1.dispersion, spectrum1.flux)
    plt.show()

    print("Tau", spectrum1.optical_depth(977))
    print("FWHM", spectrum1.fwhm(977))
    print("W", spectrum1.equivalent_width(1.03192700E+03))


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
    # fitting()
    compare()
