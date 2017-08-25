import numpy as np
from scipy.signal import savgol_filter

import astropy.units as u

from astropy.modeling import models, Parameter
from astropy.modeling.models import RedshiftScaleFactor, Linear1D
from astropy.modeling.core import Fittable1DModel

from collections import OrderedDict
import logging
import peakutils

from .utils import find_nearest
from .registries import line_registry
from .models import (TauProfile, DispersionConvert,
                     FluxConvert, FluxDecrementConvert, SmartScale, Masker)


class SpectrumModelNotImplemented(Exception): pass

class IncompleteLineInformation(Exception): pass


class SpectrumModel:
    def __init__(self, center):
        self._center = u.Quantity(center, 'Angstrom')

        self._redshift_model = RedshiftScaleFactor(0).inverse
        self._continuum_model = Linear1D(0 * u.Unit('1/Angstrom'), 1)
        self._line_model = None

        self._compound_model = None

    def __call__(self, *args, **kwargs):
        return self._compound_model(*args, **kwargs)

    def __repr__(self):
        return self._compound_model.__repr__()

    def __str__(self):
        return self._compound_model.__str__()

    def copy(self):
        new_spectrum = SpectrumModel(center=self.center.value * self.center.unit)
        new_spectrum._redshift_model = self._redshift_model.copy()
        new_spectrum._continuum_model = self._continuum_model.copy()

        if self._line_model is not None:
            new_spectrum._line_model = self._line_model.copy()

        return new_spectrum

    def _change_base(self, base):
        # Add the required input validation steps to the compound model, as
        # Astropy does not currently support adding these attributes to the
        # compound model automatically.
        # setattr(base, "input_units_strict", True)
        # setattr(base, "input_units_equivalencies", self.input_units_equivalencies)
        # setattr(base, "input_units", self.input_units)

        self._compound_model = base

        return self

    def set_redshift(self, value):
        self._redshift_model = RedshiftScaleFactor(value).inverse

        return self

    def set_continuum(self, model='Linear1D', *args, **kwargs):
        self._continum_model = getattr(models, model)(*args, **kwargs)

        return self

    def add_line(self, lambda_0=None, gamma=None, f_value=None,
                 column_density=None, v_doppler=None, delta_v=None,
                 delta_lambda=None, name=None, *args, **kwargs):
        if lambda_0 is None and name is None:
            raise IncompleteLineInformation(
                "Not enough information to construction absorption line "
                "profile. Please provide at least a name or centroid.")

        tau_prof = TauProfile(lambda_0=lambda_0, column_density=column_density,
                              v_doppler=v_doppler, gamma=gamma, f_value=f_value,
                              delta_v=delta_v, delta_lambda=delta_lambda,
                              name=name, *args, **kwargs)

        self._line_model = tau_prof if self._line_model is None else self._line_model + tau_prof

        return self

    def from_data(self, threshold=1.0, min_dist=10, strict=False,
                  interpolate=False, smooth=True, defaults=None):
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
        defaults = defaults or {}
        inv_flux = self.continuum - self.data

        # Filter with SG
        y = savgol_filter(inv_flux, 49, 3) if smooth else inv_flux

        indexes = peakutils.indexes(
            y,
            thres=threshold, # np.std(inv_flux) * self.noise/np.max(inv_flux),
            min_dist=min_dist)

        if interpolate:
            indexes = peakutils.interpolate(self.dispersion, y, ind=indexes)
            indexes = [find_nearest(self.dispersion, x) for x in indexes]

        logging.info("Found {} lines.".format(len(indexes)))

        # Add a new line to the empty spectrum object for each found line
        for ind in indexes:
            kwargs = {}
            kwargs.update(defaults)

            shifted_lambda_0 = self.dispersion[ind] * \
                               (1 + defaults['delta_v'] /
                                c.cgs) + defaults['delta_lambda']

            il = find_nearest(line_registry['wave'], shifted_lambda_0)

            nearest_wave = line_registry['wave'][il]
            nearest_name = line_registry['name'][il]
            f_value = line_registry['osc_str'][il]
            gamma_val = line_registry['gamma'][il]

            logging.info("Found {} ({}) at {}. Strict is {}.".format(
                nearest_name,
                nearest_wave,
                self.dispersion[ind],
                strict))

            # If lambda0 is provided, then we only care about the shift, not
            # the value of the line centroid
            if defaults.get('lambda_0') is not None:
                delta_v = ((shifted_lambda_0 - defaults['delta_lambda']) /
                           defaults['lambda_0'] - 1) * c.cgs
                kwargs.update({'delta_v': delta_v})

            kwargs.setdefault('lambda_0', self.dispersion[ind])

            self.add_line(**kwargs)

        return self._compound_model

    @property
    def tau(self):
        new_compound = (DispersionConvert(self._center)
                        | self._redshift_model
                        | self._line_model
                        | SmartScale(1. / (1 + self._redshift_model.z)))

        return self._change_base(new_compound)

    @property
    def flux(self):
        new_compound = (DispersionConvert(self._center)
                        | self._redshift_model
                        | (self._continuum_model
                           + (self._line_model | FluxConvert()))
                        | SmartScale(1. / (1 + self._redshift_model.z)))

        return self._change_base(new_compound)

    @property
    def flux_decrement(self):
        new_compound = (DispersionConvert(self._center)
                        | self._redshift_model
                        | (self._continuum_model
                           + (self._line_model | FluxDecrementConvert()))
                        | SmartScale(1. / (1 + self._redshift_model.z)))

        return self._change_base(new_compound)

    @property
    def masked(self):
        line_models = self._line_model if hasattr(self._line_model, '_submodels') else [self._line_model]

        mask_ranges = [line.mask_range() for line in line_models]

        return self._change_base(Masker(mask_ranges=mask_ranges) | self._compound_model)

    @property
    def line_list(self):
        return self._line_model.__repr__


