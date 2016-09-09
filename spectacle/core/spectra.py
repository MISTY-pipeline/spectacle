from .utils import find_index
from .models import Voigt1D
from .profiles import TauProfile

import os
import numpy as np
import scipy as sp
import numpy.ma as ma
import astropy.units as u
from astropy.convolution import convolve
from astropy.modeling import models, fitting
from astropy.table import Table
from uncertainties import unumpy as unp

import logging


class Spectrum1D:
    def __init__(self, dispersion=None, flux=None, uncertainty=None):
        self._uncertainty = uncertainty
        self._flux = flux
        self._dispersion = dispersion
        self._mask = None
        self._lsfs = []
        self._noise = []
        self._line_models = []
        self._remat = None
        self._model = None

        if flux is None:
            self._continuum_model = models.Linear1D(slope=0.0, intercept=1.0,
                                                    fixed={'slope': True,
                                                           'intercept': True})
        else:
            self._continuum_model = self._find_continuum()

    @property
    def model(self):
        if self._model is None:
            self._model = self._continuum_model

            if len(self._line_models) > 0:
                self._model = self._model + np.sum(self._line_models)

                # for l in [x for x in self._model.param_names if 'lambda' in x]:
                #     gamma_val = 'gamma_' + l.split('_')[-1]
                #     gamma_param = getattr(self._model, gamma_val)
                #
                #     if gamma_param.value is None:
                #         gamma_param.tied = lambda x, ln=l: \
                #             _tie_gamma(x, ln)

        return self._model

    @property
    def tau(self):
        if self.model is not None:
            return -np.log(self.model(self.dispersion))
        else:
            logging.warning("No model exists, using flux values to generate"
                            "tau.")
            return -np.log(self.flux)

    @property
    def dispersion(self):
        dispersion = self._dispersion

        if dispersion is None:
            dispersion = np.arange(0, 2000, 0.1)

        if self._remat is not None:
            dispersion = np.dot(self._remat, self._dispersion)

        return ma.masked_array(dispersion, self._mask)

    @property
    def flux(self):
        if self._flux is None:
            flux = self.model(self.dispersion)
        else:
            flux = self._flux

        # Add noise before apply lsf
        for noise in self._noise:
            flux += noise

        # Apply LSFs
        for lsf in self._lsfs:
            flux = convolve(flux, lsf.kernel)

        # Apply resampling
        if self._remat is not None:
            flux = np.dot(self._remat, flux)

        flux = ma.masked_array(flux, self._mask)

        return flux

    @flux.setter
    def flux(self, value):
        self._flux = value

    @property
    def uncertainty(self):
        if self._uncertainty is None:
            self._uncertainty = np.ones(self.flux.size)

        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, value):
        self._uncertainty = value

    @property
    def continuum(self):
        return self._continuum_model(self.dispersion)

    def _find_continuum(self, mode='LinearLSQFitter'):
        cont = models.Linear1D(slope=0.0,
                               intercept=np.median(self.flux))
        fitter = getattr(fitting, mode)()
        cont_fit = fitter(cont, self.dispersion, self.flux,
                          weights=1 / (np.abs(np.median(self.flux) -
                                              self.flux)) ** 3)

        return cont_fit

    def copy(self):
        spectrum_copy = self.__class__()
        spectrum_copy._flux = self._flux
        spectrum_copy._dispersion = self._dispersion
        spectrum_copy._mask = self._mask
        spectrum_copy._lsfs = self._lsfs
        spectrum_copy._line_models = self._line_models
        spectrum_copy._continuum_model = self._continuum_model
        spectrum_copy._remat = self._remat
        spectrum_copy._model = self._model

        return spectrum_copy

    @classmethod
    def read(cls, filename, format='fits'):
        from astropy.table import Table

        t = Table.read(filename, format)

        spectrum = cls(dispersion=t['dispersion'].data, flux=t['flux'].data,
                       uncertainty=t['uncertainty'].data)

        return spectrum

    def write(self, filename='spectrum'):
        from astropy.table import Table

        t = Table([self.dispersion, self.flux, self.uncertainty], names=(
            'dispersion', 'flux', 'uncertainty'))
        t.write(filename, format='fits')

    def set_mask(self, mask):
        if np.array(mask).shape != self._flux.shape or \
                        np.array(mask).shape != self._dispersion.shape:
            logging.warning("Mask shape does not match data shape.")
            return

        self._mask = mask

    def add_lsf(self, lsf):
        self._lsfs.append(lsf)

    def add_noise(self, std_dev=0.2):
        noise = np.random.normal(0., std_dev, self.flux.size)
        self._noise.append(noise)

        return noise

    def add_line(self, lambda_0, f_value, v_doppler, column_density,
                 gamma=None, name=""):
        model = Voigt1D(lambda_0=lambda_0, f_value=f_value, gamma=gamma or 0,
                        v_doppler=v_doppler, column_density=column_density,
                        name=name, meta={'lambda_bins': self.dispersion})

        # If gamma has not been expicitly defined, tie it to lambda
        if gamma is None:
            ind = find_index(ION_TABLE['wave'], lambda_0)
            gamma_val = ION_TABLE['gamma'][ind]
            model.gamma.value = gamma_val
            model.gamma.tied = lambda cmod, mod=model: _tie_gamma(cmod, mod)

        self._line_models.append(model)

        # Force the compound model to be recreated
        self._model = None

        return model

    def remove_model(self, model=None, x_0=None):
        if model is not None:
            self._line_models.remove(model)
        elif x_0 is not None:
            model = self.get_profile(x_0)
            self._line_models.remove(model)

    def set_continuum(self, function, *args, **kwargs):
        model = getattr(models, function)
        self._continuum_model = model(*args, **kwargs)

    def get_profile(self, x_0):
        # Find the nearest voigt profile to the given central wavelength
        v_arr = sorted(self._line_models, key=lambda x: x.lambda_0.value)
        v_x_0_arr = [x.lambda_0.value for x in v_arr]

        if len(v_x_0_arr) > 1:
            ind = find_index(v_x_0_arr, x_0)

            # Retrieve the voigt profile at that wavelength
            v_prof = v_arr[ind]
        else:
            v_prof = v_arr[0]

        return v_prof

    def fwhm(self, x_0):
        """
        Calculates an approximation of the FWHM.

        The approximation is accurate to
        about 0.03% (see http://en.wikipedia.org/wiki/Voigt_profile).

        Returns
        -------
        FWHM : float
            The estimate of the FWHM
        """
        v_prof = self.get_profile(x_0)

        # The width of the Lorentz profile
        fl = 2.0 * v_prof.gamma

        # Width of the Gaussian [2.35 = 2*sigma*sqrt(2*ln(2))]
        fd = 2.35482 * 1/np.sqrt(2.)

        return 0.5346 * fl + np.sqrt(0.2166 * (fl ** 2.) + fd ** 2.)

    def optical_depth(self, x_0):
        """
        Return the optical depth at some wavelength.

        Parameters
        ----------
        x_0 : float
            Line center from which to calculate tau.

        Returns
        -------
        tau : float
            The value of the optical depth at the given wavelength.
        """
        idx = (np.abs(self.dispersion - x_0)).argmin()
        return -np.log(self.flux[idx])

    def centroid(self, x_0):
        """
        Return the centroid for Voigt profile near the given wavelength.

        Parameters
        ----------
        x_0 : float
            Wavelength new the given profile from which to calculate the
            centroid.

        Returns
        -------
        cent : float
            The centroid of the profile.
        """
        profile = self.get_profile(x_0)
        disp = self.dispersion
        flux = profile(disp)

        cent = np.trapz(disp * flux, disp) / np.trapz(flux, disp)

        return cent

    def equivalent_width(self, x1=None, x2=None):
        if x1 is None:
            x1 = self.dispersion[0]

        if x2 is None:
            x2 = self.dispersion[-1]

        mask = (self.dispersion >= x1) & (self.dispersion <= x2)
        disp = self.dispersion[mask]
        flux = self.flux[mask]
        uncert = self.uncertainty[mask]

        # Compose the uncertainty array
        uflux = unp.uarray(flux, uncert)

        # Continuum is always assumed to be 1.0
        avg_cont = 1.0

        # Average dispersion in the line region.
        avg_dx = np.mean(disp[1:] - disp[:-1])

        # Calculate equivalent width
        ew = ((avg_cont - uflux) * (avg_dx / avg_cont)).sum()

        return ew.nominal_value, ew.std_dev

    def resample(self, dispersion):
        remat = self._resample_matrix(self.dispersion, dispersion)
        self._remat = remat

    def _resample_matrix(self, orig_lamb, fin_lamb):
        """
        Create a resampling matrix to be used in resampling spectra in a way
        that conserves flux. This is adapted from code created by the SEAGal
        Group.

        .. note:: This method assumes uniform grids.

        Parameters
        ----------
        orig_lamb : ndarray
            The original dispersion array.
        fin_lamb : ndarray
            The desired dispersion array.

        Returns
        -------
        resample_map : ndarray
            An [[N_{fin_lamb}, M_{orig_lamb}]] matrix.
        """
        # Get step size
        delta_orig = orig_lamb[1] - orig_lamb[0]
        delta_fin = fin_lamb[1] - fin_lamb[0]

        n_orig_lamb = len(orig_lamb)
        n_fin_lamb = len(fin_lamb)

        # Lower bin and upper bin edges
        orig_low = orig_lamb - delta_orig * 0.5
        orig_upp = orig_lamb + delta_orig * 0.5
        fin_low = fin_lamb - delta_fin * 0.5
        fin_upp = fin_lamb + delta_fin * 0.5

        # Create resampling matrix
        resamp_mat = np.zeros(shape=(n_fin_lamb, n_orig_lamb))

        for i in range(n_fin_lamb):
            # Calculate the contribution of each original bin to the
            # resampled bin
            l_inf = np.where(orig_low > fin_low[i], orig_low, fin_low[i])
            l_sup = np.where(orig_upp < fin_upp[i], orig_upp, fin_upp[i])

            # Interval overlap of each original bin for current resampled
            # bin; negatives clipped
            dl = (l_sup - l_inf).clip(0)

            # This will only happen at the edges of lorig.
            # Discard resampled bin if it's not fully covered (> 99%) by the
            #  original bin -- only happens at the edges of the original bins
            if 0 < dl.sum() < 0.99 * delta_fin:
                dl = 0 * orig_lamb

            resamp_mat[i, :] = dl

        resamp_mat /= delta_fin

        return resamp_mat


ION_TABLE = Table.read(
    os.path.abspath(
        os.path.join(__file__, '..', '..', 'data', 'line_list', 'atom.dat')),
    format='ascii', names=('name', 'wave', 'osc_str', 'gamma'))


def _tie_gamma(compound_model, model):
    # Find the index of the original model in the compound model
    mod_ind = compound_model._submodels.index(model)

    # The auto-generated name of the parameter in the compoound model
    param_name = "lambda_0_{}".format(mod_ind)
    lambda_val = getattr(compound_model, param_name).value

    ind = find_index(ION_TABLE['wave'], lambda_val)
    gamma_val = ION_TABLE['gamma'][ind]

    return gamma_val
