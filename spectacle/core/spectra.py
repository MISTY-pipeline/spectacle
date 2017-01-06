from .utils import find_index, ION_TABLE
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
                 dispersion_unit=None):
        if dispersion is None:
            self._dispersion = np.linspace(0, 2000, len(data))
        else:
            self._dispersion = dispersion

        if uncertainty is None:
            uncertainty = np.ones(len(data))

        self._dispersion_unit = None
        self._lsfs = []
        self._noise = []
        self._remat = None

        super(Spectrum1D, self).__init__(data, uncertainty=uncertainty)

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

    def resample(self, dispersion, copy=True, **kwargs):
        remat = resample(self.dispersion, dispersion, **kwargs)

        if copy:
            new_spec = self.copy()
            new_spec._remat = remat
        else:
            self._remat = remat
            new_spec = self

        return new_spec
