from spectacle.core.spectra import Spectrum1D
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from spectacle.modeling.fitting import Fitter
from astropy.modeling.models import Linear1D
from spectacle.core.models import Voigt1D


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
    spectrum2 = Spectrum1D()

    for i, n in enumerate(np.logspace(7, 25, 20)):
        spectrum2.add_line(lambda_0=1.03192700E+03 * (i+1), f_value=0.4164,
                           gamma=1e5,
                           v_doppler=1e6, column_density=n)
        print(np.log10(n), np.log10(spectrum2.equivalent_width(
            1.03192700E+03)))
        log_n.append(np.log10(n * 0.4164 * (1.03192700E+03 / 5000.0)))
        log_w.append(np.log10(spectrum2.equivalent_width(1.03192700E+03) /
                              1.03192700E+03))
    print(spectrum2.model)
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
    # hdulist = fits.open("/Users/nearl/projects/hst_proposal/QSOALS/3C066A/3C066A_coadd_FUVM_final_all.fits")
    # disp, flux = hdulist[1].data['WAVE'], hdulist[1].data['FLUX']
    #
    # flux = flux[(disp > 1350) & (disp < 1400)]
    # flux /= np.median(flux)
    # disp = disp[(disp > 1350) & (disp < 1400)]
    #
    # spectrum = Spectrum1D(disp, flux)
    spectrum = Spectrum1D()
    spectrum.add_line(lambda_0=1.03192700E+03, f_value=0.4164, gamma=1e6,
                      v_doppler=1e7, column_density=10 ** 14.66)
    spectrum.add_line(lambda_0=1.13192700E+03, f_value=0.6164, gamma=1e6,
                      v_doppler=1e7, column_density=10 ** 13.55)
    spectrum.add_noise()
    spectrum.uncertainty = (np.random.sample(
        spectrum.dispersion.size)) * 0.01

    fitter = Fitter()
    result_spectrum = fitter(spectrum)

    plt.plot(spectrum.dispersion, spectrum.flux)
    # plt.plot(spectrum.dispersion, spectrum.uncertainty, marker='o')
    plt.plot(result_spectrum.dispersion, result_spectrum.flux)
    plt.show()

    # new_spectrum = Spectrum1D()
    # print(result_spectrum.model.parameters[3:8])
    # new_spectrum.add_line(*result_spectrum.model.parameters[3:8])
    #
    # plt.plot(spectrum.dispersion, spectrum.flux)
    # # plt.plot(spectrum.dispersion, spectrum.uncertainty, marker='o')
    # plt.plot(new_spectrum.dispersion, new_spectrum.flux)
    # plt.show()


def simple():
    spectrum = Spectrum1D()
    spectrum.add_line(lambda_0=1.03192700E+03, f_value=0.4164, gamma=1e6,
                      v_doppler=1e7, column_density=10 ** 14.66)
    spectrum.add_line(lambda_0=1.13192700E+03, f_value=0.4164, gamma=1e6,
                      v_doppler=1e7, column_density=10 ** 14.66)

    plt.plot(spectrum.dispersion, spectrum.flux)
    plt.show()


if __name__ == '__main__':
    fitting()
    # compare()
    # simple()