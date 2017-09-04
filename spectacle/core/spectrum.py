import astropy.units as u

from astropy.modeling import models
from astropy.modeling.models import Linear1D

from ..models.profiles import TauProfile
from ..models.custom import SmartScale, Redshift
from ..models.converters import (DispersionConvert, FluxConvert,
                                 FluxDecrementConvert)


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

    @property
    def center(self):
        return self._center

    def copy(self):
        new_spectrum = Spectrum1D(center=self.center.value * self.center.unit)
        new_spectrum._redshift_model = self._redshift_model.copy()
        new_spectrum._continuum_model = self._continuum_model.copy()

        if self._line_model is not None:
            new_spectrum._line_model = self._line_model.copy()

        return new_spectrum

    def set_redshift(self, *args, **kwargs):
        self._redshift_model = Redshift(*args, **kwargs)

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
        ss = SmartScale(1. / (1 + self._redshift_model.z))
        lm = self._line_model

        comp_mod = (dc | rs | ss | lm) if lm is not None else (dc | rs | ss)

        return comp_mod

    @property
    def flux(self):
        dc = DispersionConvert(self._center)
        rs = self._redshift_model
        ss = SmartScale(1. / (1 + self._redshift_model.z))
        cm = self._continuum_model
        lm = self._line_model
        fc = FluxConvert()

        comp_mod = (dc | rs | ss | (cm + (lm | fc))) if lm is not None else (dc | rs | ss | cm)

        comp_mod.input_units = dc.input_units

        return comp_mod

    @property
    def flux_decrement(self):
        dc = DispersionConvert(self._center)
        rs = self._redshift_model
        cm = self._continuum_model
        lm = self._line_model
        fd = FluxDecrementConvert()
        ss = SmartScale(1. / (1 + self._redshift_model.z))

        comp_mod = (dc | rs | ss | (cm + (lm | fd))) if lm is not None else (dc | rs | ss | cm)

        return comp_mod

    @property
    def masked(self):
        line_models = self._line_model if hasattr(self._line_model, '_submodels') else [self._line_model]

        mask_ranges = [line.mask_range() for line in line_models]

        return self._change_base(Masker(mask_ranges=mask_ranges) | self._compound_model)

    @property
    def line_list(self):
        return self._line_model.__repr__


