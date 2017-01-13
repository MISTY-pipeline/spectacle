from .utils import find_nearest, ION_TABLE, find_bounds
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
    """
    A spectrum container object for real or synthetic spectral data.
    """
    def __init__(self, data, dispersion=None, uncertainty=None,
                 dispersion_unit=None, *args, **kwargs):
        if dispersion is None:
            self._dispersion = np.linspace(0, 2000, len(data))
        else:
            self._dispersion = dispersion

        if uncertainty is None:
            uncertainty = StdDevUncertainty(np.zeros(len(data)))

        self._dispersion_unit = None
        self._lsfs = []
        self._noise = []
        self._remat = None

        super(Spectrum1D, self).__init__(data, uncertainty=uncertainty, *args,
                                         **kwargs)

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
        data = self._process()

        return data

    def _process(self):
        data = np.copy(self._data)

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

        center = x_0 or self.dispersion[0]
        center = center * u.Angstrom
        dispersion = self.dispersion[mask] * u.Angstrom

        velocity = ((dispersion - center) /
                    dispersion * c.c.cgs).to("km/s").value

        return velocity

    @NDDataRef.uncertainty.getter
    def uncertainty(self):
        # The uncertainty array object is an NDUncertainty class; retrieve just
        # the array values from it
        uncert = self._uncertainty.array

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

    def copy(self, deep_copy=True, **kwargs):
        """
        Create a new `Spectrum1D` object using current property
        values. Takes all the arguments a Spectrum1D expects, arguments that
        are not included use this instance's values.
        """
        self_kwargs = {"data": self.data,
                       "dispersion": self.dispersion,
                       "unit": self.unit, "wcs": self.wcs,
                       "uncertainty": self.uncertainty,
                       "mask": self.mask, "meta": self.meta}

        self_kwargs.update(kwargs)

        return self.__class__(**self_kwargs)

    def add_lsf(self, lsf):
        self._lsfs.append(lsf)

    def add_noise(self, std_dev=0.2):
        if std_dev > 0:
            noise = np.random.normal(0., std_dev, self.data.size)
            self._noise.append(noise)

            return noise

    def resample(self, dispersion, copy=True, **kwargs):
        remat = resample(self.dispersion, dispersion, **kwargs)

        if copy:
            new_spec = self.copy()
            new_spec._remat = remat
        else:
            self._remat = remat
            new_spec = self

        return new_spec

    def _get_range_mask(self, x_0):
        # This makes the assumption that the continuum has been normalized to 1
        cont = np.ones(self.dispersion.shape)
        diff = np.isclose(self.data, cont, rtol=1e-2, atol=1e-5)

        inds = np.where(diff)[0]

        return inds

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
        idx = find_nearest(self.data, x_0)
        tau = unp.log(1.0/flux[idx])

        return unp.nominal_values(tau), unp.std_devs(tau)

    def centroid(self, x_0=None, x_range=None, line_name=None):
        """
        Return the centroid for Voigt profile near the given wavelength.

        Parameters
        ----------
        x_0 : float, optional
            Wavelength new the given profile from which to calculate the
            centroid.
        x_range : list-like, optional
            A list or tuple of size 2 which define the start and end
            wavelengths demarcating the range of interest.
        line_name : string, optional
            This will trigger a look-up of the `x_0` value in the provided ion
            table.

        Returns
        -------
        float, float
            The centroid of the profile, and the associated uncertainty.
        """
        if x_range is not None and (isinstance(x_range, list) or
                                        isinstance(x_range, tuple)):
            x1, x2 = x_range
        elif x_0 is not None:
            ind_x = find_nearest(self.dispersion, x_0)
            ind_left, ind_right = find_bounds(self.data, ind_x, 1.0, cap=True)
            x1, x2 = self.dispersion[ind_left], self.dispersion[ind_right]
        else:
            x1, x2 = self.dispersion[0], self.dispersion[-1]

        mask = (self.dispersion >= x1) & (self.dispersion <= x2)
        disp = self.dispersion[mask]
        flux = self.data[mask]
        uncert = self.uncertainty[mask]

        cent = np.trapz(disp * flux, disp) / np.trapz(flux, disp)

        return unp.nominal_values(cent), unp.std_devs(cent)

    def equivalent_width(self, x_0=None, x_range=None, line_name=None):
        """
        Return the centroid for Voigt profile near the given wavelength.

        Parameters
        ----------
        x_0 : float, optional
            Wavelength new the given profile from which to calculate the
            centroid.
        x_range : list-like, optional
            A list or tuple of size 2 which define the start and end
            wavelengths demarcating the range of interest.
        line_name : string, optional
            This will trigger a look-up of the `x_0` value in the provided ion
            table.

        Returns
        -------
        float, float
            The equivalent width of the profile, and the associated
            uncertainty.
        """
        if x_range is not None and (isinstance(x_range, list) or
                                    isinstance(x_range, tuple)):
            x1, x2 = x_range
        elif x_0 is not None:
            ind_x = find_nearest(self.dispersion, x_0)
            # TODO: try applying a smoothing kernel before calculating bounds
            ind_left, ind_right = find_bounds(self.data, ind_x, 1.0, cap=True)
            x1, x2 = self.dispersion[ind_left], self.dispersion[ind_right]
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
