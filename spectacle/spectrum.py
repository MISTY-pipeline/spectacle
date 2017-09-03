import numpy as np
from scipy.signal import savgol_filter

import astropy.units as u
from astropy.constants import c

from astropy.modeling import models, Parameter
from astropy.modeling.models import RedshiftScaleFactor, Linear1D
from astropy.modeling.core import Fittable1DModel

from collections import OrderedDict
import logging
import peakutils

from .utils import find_nearest
from .registries import line_registry
from .models import (TauProfile, DispersionConvert,
                     FluxConvert, FluxDecrementConvert, SmartScale, Masker, Redshift)


class SpectrumModelNotImplemented(Exception): pass

class InappropriateModel(Exception): pass

class NoLines(Exception): pass


class Spectrum1D:
    def __init__(self, center=0, z=0):
        self._center = u.Quantity(center, 'Angstrom')

        self._redshift_model = Redshift(z).inverse
        self._continuum_model = Linear1D(0 * u.Unit('1/Angstrom'), 1)
        self._line_model = None

        self._compound_model = None

    def __call__(self):
        raise InappropriateModel("This is not a model object; choose either"
                                 "tau, flux, or flux decrement properties.")

    def __repr__(self):
        return self._compound_model.__repr__()

    def __str__(self):
        return self._compound_model.__str__()

    def copy(self):
        new_spectrum = Spectrum1D(center=self.center.value * self.center.unit)
        new_spectrum._redshift_model = self._redshift_model.copy()
        new_spectrum._continuum_model = self._continuum_model.copy()

        if self._line_model is not None:
            new_spectrum._line_model = self._line_model.copy()

        return new_spectrum

    def set_redshift(self, value):
        self._redshift_model.z = value

        return self

    def set_continuum(self, model='Linear1D', *args, **kwargs):
        self._continum_model = getattr(models, model)(*args, **kwargs)

        return self

    def add_line(self, *args, **kwargs):
        tau_prof = TauProfile(*args, **kwargs)

        self._line_model = tau_prof if self._line_model is None \
                                    else self._line_model + tau_prof

        return self

    @property
    def tau(self):
        dc = DispersionConvert(self._center)
        rs = self._redshift_model
        lm = self._line_model
        ss = SmartScale(1. / (1 + self._redshift_model.z))

        return (dc | rs | lm | ss) if lm is not None else (dc | rs | ss)

    @property
    def flux(self):
        dc = DispersionConvert(self._center)
        rs = self._redshift_model
        cm = self._continuum_model
        lm = self._line_model
        fc = FluxConvert()
        ss = SmartScale(1. / (1 + self._redshift_model.z))

        return (dc | rs | (cm + (lm | fc)) | ss) if lm is not None else (dc | rs | cm | ss)

    @property
    def flux_decrement(self):
        dc = DispersionConvert(self._center)
        rs = self._redshift_model
        cm = self._continuum_model
        lm = self._line_model
        fd = FluxDecrementConvert()
        ss = SmartScale(1. / (1 + self._redshift_model.z))

        return (dc | rs | (cm + (lm | fd)) | ss) if lm is not None else (dc | rs | cm | ss)

    @property
    def masked(self):
        line_models = self._line_model if hasattr(self._line_model, '_submodels') else [self._line_model]

        mask_ranges = [line.mask_range() for line in line_models]

        return self._change_base(Masker(mask_ranges=mask_ranges) | self._compound_model)

    @property
    def line_list(self):
        return self._line_model.__repr__


