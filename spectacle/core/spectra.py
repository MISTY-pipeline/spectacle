from .utils import find_index, ION_TABLE
from .models import Voigt1D
from ..analysis.resample import resample

import numpy as np
from astropy import constants as c
from astropy import units as u
from astropy.convolution import convolve
from astropy.modeling import models, fitting
from astropy.nddata import NDDataRef, NDUncertainty, StdDevUncertainty
from astropy.wcs import WCS, WCSSUB_SPECTRAL
from uncertainties import unumpy as unp

import logging


class Spectrum1D(NDDataRef):
    def __init__(self, data, dispersion=None, uncertainty=None,
                 dispersion_unit=None, **kwargs):
        # If the uncertainty value is not an NDUncertainty class, make it so
        if not isinstance(uncertainty, NDUncertainty):
            if uncertainty is None:
                # Provide an uncertainty array with no value
                uncertainty = np.zeros(data.size)

            # We assume the uncertainties are standard deviations
            uncertainty = StdDevUncertainty(uncertainty)

        super(Spectrum1D, self).__init__(data, uncertainty=uncertainty,
                                         **kwargs)

        self._dispersion = dispersion
        self._dispersion_unit = None
        self._lsfs = []
        self._noise = []
        self._line_models = []
        self._remat = None
        self._model = None

        self._continuum_model = self._find_continuum()

    def __repr__(self):
        return self.model

    @property
    def model(self):
        """
        Returns the complete :class:`astropy.modeling.Models` object.
        """
        if self._model is None:
            self._model = self._continuum_model

            if len(self._line_models) > 0:
                self._model = self._model + np.sum(self._line_models)

        return self._model

    @property
    def data(self):
        """
        Returns the calculated flux of the spectrum. This is either

        1. Defined explicitly via some real or synthetic data, or
        2. Calculated based on a set of models representing the continuum and
           any associated absorption features.

        In all cases, any noise defined by the user will be added, as well as
        any line spread functions or resampling.

        Returns
        -------
        data : ndarray
            The resulting data array representing the flux of the spectrum.

        """
        if self._data is None:
            data = self.model(self.dispersion)
        else:
            data = self._data

        # Add noise before apply lsf
        for noise in self._noise:
            data += noise

        # Apply LSFs
        for lsf in self._lsfs:
            # Convolving has unintended effects on the ends of the spectrum.
            # Use tau instead.
            tau = np.log(1/data)
            tau = convolve(tau, lsf.kernel)
            data = np.exp(-tau)

        # Apply resampling
        if self._remat is not None:
            data = np.dot(self._remat, data)

        # Flux values cannot be negative
        data[data < 0.0] = 0.0

        return data

    @property
    def dispersion(self):
        """
        The dispersion axis of the spectral object. This property will return a
        default dispersion if no custom dispersion has been provided. The
        returned dispersion object will also be rebinned if a binning matrix
        has been calculated.

        Returns
        -------
        dispersion : ndarray
            The spectral dispersion values.
        """
        # If dispersion has not yet been defined, attempt to use the wcs
        # information, if it exists
        if self._dispersion is None and self.wcs is not None:
            self._dispersion = np.arange(self.data.shape[0])

            if isinstance(self.wcs, WCS):
                # Try to reference the spectral axis
                wcs_spec = self.wcs.sub([WCSSUB_SPECTRAL])

                # Check to see if it actually is a real coordinate description
                if wcs_spec.naxis == 0:
                    # It's not real, so attempt to get the spectral axis by
                    # specifying axis by integer
                    wcs_spec = self.wcs.sub([self.wcs.naxis])

                # Construct the dispersion array
                self._dispersion = wcs_spec.all_pix2world(
                    np.arange(self.data.shape[0]), 0)[0]

        dispersion = self._dispersion

        # If dispersion is still none, create a basic wavelength array
        if dispersion is None:
            dispersion = np.linspace(0, 2000, self.data.size)

        if self._remat is not None:
            dispersion = np.dot(self._remat, dispersion)

        return dispersion

    @property
    def dispersion_unit(self):
        # If wcs information is provided, attempt to get the dispersion unit
        # from the header
        if self._dispersion_unit is None and self.wcs is not None:
            try:
                self._dispersion_unit = self.wcs.wcs.cunit[0]
            except AttributeError:
                logging.warning("No dispersion unit information in WCS.")

                try:
                    self._dispersion_unit = u.Unit(
                        self.meta['header']['cunit'][0])
                except KeyError:
                    logging.warning("No dispersion unit information in meta.")

                    self._dispersion_unit = u.Unit("")

        return self._dispersion_unit

    @dispersion_unit.setter
    def dispersion_unit(self, value):
        self._dispersion_unit = value

    def velocity(self, x_0=None, mask=None):
        """
        Calculates the velocity values of the dispersion axis.

        Parameters
        ----------
        x_0 : float
            Lambda value of the center wavelength element.
        mask : ndarray
            Boolean array describing which elements are to be included in the
            returned velocity array.

        Returns
        -------
        velocity : ndarray
            Velocity values given the central wavelength and dispersion
            information.
        """
        mask = mask if mask is not None else np.ones(
            shape=self.dispersion.shape, dtype=bool)
        center = x_0 or self.get_profile(0.0).lambda_0
        dispersion = self.dispersion[mask] * u.Angstrom
        center = center * u.Angstrom
        velocity = ((dispersion - center) /
                    dispersion * c.c.cgs).to("km/s").value

        return velocity

    @NDDataRef.uncertainty.getter
    def uncertainty(self):
        uncert = self._uncertainty

        # Apply LSFs
        for lsf in self._lsfs:
            # Convolving has unintended effects on the ends of the spectrum.
            # Use tau instead.
            tau = np.log(1/uncert)
            tau = convolve(tau, lsf.kernel)
            uncert = np.exp(-tau)

        # Apply resampling
        if self._remat is not None:
            uncert = np.dot(self._remat, uncert)

        return uncert

    @property
    def tau(self):
        tau = unp.log(1.0 / unp.uarray(self.data, self.uncertainty))

        return unp.nominal_values(tau)

    @property
    def tau_uncertainty(self):
        tau = unp.log(1.0 / unp.uarray(self.data, self.uncertainty))

        return unp.std_devs(tau)

    @property
    def continuum(self):
        return self._continuum_model(self.dispersion)

    @property
    def line_list(self):
        """
        List all available line names.
        """
        return ION_TABLE

    def _find_continuum(self, mode='LinearLSQFitter'):
        cont = models.Linear1D(slope=0.0,
                               intercept=np.median(self.data))
        fitter = getattr(fitting, mode)()
        cont_fit = fitter(cont, self.dispersion, self.data,
                          weights=np.abs(np.median(self.data) -
                                         self.data) ** -3)

        return cont_fit

    @classmethod
    def copy(cls, original, deep_copy=True, **kwargs):
        """
        Create a new `Spectrum1D` object using current property
        values. Takes all the arguments a Spectrum1D expects, arguments that
        are not included use this instance's values.
        """
        self_kwargs = {"data": original._data,
                       "dispersion": original._dispersion,
                       "unit": original.unit, "wcs": original.wcs,
                       "uncertainty": original._uncertainty,
                       "mask": original.mask, "meta": original.meta}

        self_kwargs.update(kwargs)

        return cls(copy=deep_copy, **self_kwargs)

    def add_lsf(self, lsf):
        self._lsfs.append(lsf)

    def add_noise(self, std_dev=0.2):
        if std_dev > 0:
            noise = np.random.normal(0., std_dev, self.data.size)
            self._noise.append(noise)

            return noise

    def add_line(self, v_doppler, column_density, lambda_0=None, f_value=None,
                 gamma=None, delta_v=None, delta_lambda=None, name=None):
        if name is not None:
            ind = np.where(ION_TABLE['name'] == name)
            lambda_0 = ION_TABLE['wave'][ind]
        else:
            ind = find_index(ION_TABLE['wave'], lambda_0)
            name = ION_TABLE['name'][ind]

        if f_value is None:
            f_value = ION_TABLE['osc_str'][ind]

        model = Voigt1D(lambda_0=lambda_0, f_value=f_value, gamma=gamma or 0,
                        v_doppler=v_doppler, column_density=column_density,
                        delta_v=delta_v, delta_lambda=delta_lambda, name=name,
                        meta={'lambda_bins': self.dispersion}
                        )

        # If gamma has not been explicitly defined, tie it to lambda
        if gamma is None:
            gamma_val = ION_TABLE['gamma'][ind]
            model.gamma.value = gamma_val
            model.gamma.tied = lambda cmod, mod=model: _tie_gamma(cmod, mod)

        self._line_models.append(model)

        # Force the compound model to be recreated
        self._model = None

        # Set the new dispersion value
        self._dispersion = model.lambda_bins

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
        v_x_0_arr = np.array([x.lambda_0.value for x in v_arr])

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
        flux = unp.uarray(self.data, self.uncertainty)
        idx = (np.abs(self.dispersion - x_0)).argmin()
        tau = unp.log(1.0/flux[idx])

        return unp.nominal_values(tau), unp.std_devs(tau)

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
        disp = self.dispersion
        flux = unp.uarray(self.data, self.uncertainty)

        cent = np.trapz(disp * flux, disp) / np.trapz(flux, disp)

        return unp.nominal_values(cent), unp.std_devs(cent)

    def equivalent_width(self, x_range=None, x_0=None, line_name=None):
        if x_range is not None and (isinstance(x_range, list) or
                                    isinstance(x_range, tuple)):
            x1, x2 = x_range
        elif x_0 is not None or line_name is not None:
            region_mask = self._get_range_mask(x_0)
            region_disp = self.dispersion[region_mask]
            x1, x2 = region_disp[0], region_disp[-1]
        else:
            x1, x2 = self.dispersion[0], self.dispersion[-1]

        mask = (self.dispersion >= x1) & (self.dispersion <= x2)
        disp = self.dispersion[mask]
        flux = self.data[mask]
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

    def _get_range_mask(self, x_0=None):
        profile = np.sum(self._line_models) #self.get_profile(x_0 or 0.0)
        vdisp = profile(self.dispersion)
        cont = np.zeros(self.dispersion.shape)

        return ~np.isclose(vdisp, cont, rtol=1e-2, atol=1e-5)

    def resample(self, dispersion, copy=True, **kwargs):
        remat = resample(self.dispersion, dispersion, **kwargs)

        if copy:
            new_spec = self.copy()
            new_spec._remat = remat
        else:
            self._remat = remat
            new_spec = self

        return new_spec


def _tie_gamma(compound_model, model):
    # Find the index of the original model in the compound model
    mod_ind = compound_model._submodels.index(model)

    # The auto-generated name of the parameter in the compound model
    param_name = "lambda_0_{}".format(mod_ind)
    lambda_val = getattr(compound_model, param_name).value

    ind = find_index(ION_TABLE['wave'], lambda_val)
    gamma_val = ION_TABLE['gamma'][ind]

    return gamma_val
