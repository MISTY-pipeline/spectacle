from .utils import find_nearest, find_bounds
from ..analysis.resample import resample
from .registries import line_registry
from .lines import Line

import numpy as np
from astropy import constants as c
from astropy import units as u
from astropy.convolution import convolve
from astropy.nddata import NDDataRef, StdDevUncertainty
from astropy.wcs import WCS, WCSSUB_SPECTRAL
from uncertainties import unumpy as unp
from scipy.signal import savgol_filter
import peakutils

import logging


class Spectrum1D(NDDataRef):
    """
    A spectrum container object for real or synthetic spectral data.
    """
    def __init__(self, data, dispersion=None, uncertainty=None,
                 dispersion_unit=None, lines=None, tau=None, *args, **kwargs):
        if dispersion is None:
            self._dispersion = np.linspace(0, 2000, len(data))
        else:
            self._dispersion = dispersion

        if uncertainty is None:
            uncertainty = StdDevUncertainty(np.zeros(len(data)))

        self._dispersion_unit = dispersion_unit
        self._lsfs = []
        self._noise = []
        self._remat = None
        self._lines = lines if lines is not None else {}
        self._tau = tau

        super(Spectrum1D, self).__init__(data, uncertainty=uncertainty, *args,
                                         **kwargs)

    @classmethod
    def formats(cls):
        """
        Retrieves the currently loaded formats available for reading/writing
        spectral data.

        Returns
        -------
        :class:`astropy.table.Table`
            An Astropy Table object listing the available formats for the
            :class:`spectacle.core.spectra.Spectrum1D` object.
        """
        from astropy.io import registry as io_registry

        io_registry.get_formats(cls)

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

    @uncertainty.setter
    def uncertainty(self, value):
        value = StdDevUncertainty(value)
        NDDataRef.uncertainty.fset(self, value)

    @property
    def tau(self):
        if self._tau is None:
            tau = unp.log(1.0 / unp.uarray(self.data, self.uncertainty))

            return unp.nominal_values(tau)

        return self._tau

    @property
    def tau_uncertainty(self):
        if self._tau is None:
            tau = unp.log(1.0 / unp.uarray(self.data, self.uncertainty))

            return unp.std_devs(tau)

    @property
    def lines(self):
        return self._lines

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

    def _get_line_mask(self, x_0):
        # TODO: try applying a smoothing kernel before calculating bounds
        ind_left, ind_right = find_bounds(self.dispersion, self.data, x_0, 1.0,
                                          cap=True)
        x1, x2 = self.dispersion[ind_left], self.dispersion[ind_right]

        mask = (self.dispersion >= x1) & (self.dispersion <= x2)

        return mask

    @property
    def line_mask(self):
        mask_list = []

        for lambda_0 in self._lines.values():
            mask_list.append(self._get_line_mask(lambda_0))

        if len(mask_list) == 0:
            line_mask = np.zeros(shape=self.dispersion.shape, dtype=bool)
        else:
            line_mask = np.logical_or.reduce(mask_list)

        return line_mask

    def find_lines(self, threshold=0.7, min_dist=100, strict=False):
        """
        Simple peak finder.

        Parameters
        ----------
        self : :class:`spectacle.core.spectra.Spectrum1D`
            Original spectrum object.
        strict : bool
            If `True`, will require that any found peaks also exist in the
            provided ions lookup table.

        Returns
        -------
        indexes : np.array
            An array of indices providing the peak locations in the original
            spectrum object.
        """
        continuum = np.median(self.data)
        inv_flux = 1 - self.data

        # Filter with SG
        y = savgol_filter(inv_flux, 49, 3)

        indexes = peakutils.indexes(
            y,
            thres=threshold, # np.std(inv_flux) * self.noise/np.max(inv_flux),
            min_dist=min_dist)

        if strict:
            print("Using strict line associations.")
            # Find the indices of the ion table that correspond with the found
            # indices in the peak search
            tab_indexes = np.array(list(set(
                map(lambda x: find_nearest(line_registry['wave'], x),
                    self.dispersion[indexes]))))

            # Given the indices in the ion tab, find the indices in the
            #  dispersion array that correspond to the ion table lambda
            indexes = np.array(list(
                map(lambda x: find_nearest(self.dispersion, x),
                    line_registry['wave'][tab_indexes])))

        logging.info("Found {} lines.".format(len(indexes)))

        line_list = {}

        # Add a new line to the empty spectrum object for each found line
        for ind in indexes:
            il = find_nearest(line_registry['wave'],
                              self.dispersion[ind])

            nearest_wave = line_registry['wave'][il]
            nearest_name = line_registry['name'][il]
            f_value = line_registry['osc_str'][il]
            gamma_val = line_registry['gamma'][il]

            if nearest_name in line_list:
                nearest_name = "{}_{}".format(
                    nearest_name,
                    len([k for k in line_list if k == nearest_name]))

            logging.info("Found {} ({}) at {}. Strict is {}.".format(
                nearest_name,
                nearest_wave,
                self.dispersion[ind],
                strict))

            mod = Line(lambda_0=self.dispersion[ind], f_value=f_value,
                       gamma=gamma_val, v_doppler=1e6, column_density=10**14,
                       name=nearest_name)

            if mod is not None:
                line_list[nearest_name] = mod

        logging.info("There are {} lines in the model.".format(
            len(line_list)))

        return list(line_list.values())

    def optical_depth(self):
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
        tau = unp.log(1.0/flux)

        return unp.nominal_values(tau), unp.std_devs(tau)

    def centroid(self, x_0=None, x_range=None):
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

        .. warning:: Currently, uncertainties are not included in the
                     calculation.

        Returns
        -------
        float, float
            The centroid of the profile, and the associated uncertainty.
        """
        if x_range is not None and (isinstance(x_range, list) or
                                    isinstance(x_range, tuple)):
            mask = (self.dispersion >= x_range[0]) & \
                   (self.dispersion <= x_range[1])
        elif x_0 is not None:
            mask = self._get_line_mask(x_0)
        else:
            mask = (self.dispersion >= self.dispersion[0]) & \
                   (self.dispersion <= self.dispersion[1])

        disp = self.dispersion[mask]
        flux = self.data[mask]
        uncert = self.uncertainty[mask]

        cent = np.trapz(disp * flux, disp) / np.trapz(flux, disp)

        return unp.nominal_values(cent), unp.std_devs(cent)

    def equivalent_width(self, x_0=None, x_range=None):
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
            mask = (self.dispersion >= x_range[0]) & \
                   (self.dispersion <= x_range[1])
        elif x_0 is not None:
            mask = self._get_line_mask(x_0)
        else:
            mask = (self.dispersion >= self.dispersion[0]) & \
                   (self.dispersion <= self.dispersion[1])

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