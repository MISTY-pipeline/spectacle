import numpy as np

import astropy.units as u

from astropy.modeling import models, Parameter
from astropy.modeling.models import RedshiftScaleFactor, Linear1D, Scale
from astropy.modeling.core import _CompoundModel, Fittable1DModel

from .models import (TauProfile, WavelengthConvert, VelocityConvert,
                     FluxConvert, FluxDecrementConvert)


class SpectrumModelNotImplemented(Exception): pass

class IncompleteLineInformation(Exception): pass


class Spectrum1D(Fittable1DModel):
    num_lines = Parameter(default=0)

    def __init__(self, *args, **kwargs):
        super(Spectrum1D, self).__init__(*args, **kwargs)
        self._redshift_model = RedshiftScaleFactor(0)
        self._continuum_model = Linear1D(0, 1)
        self._line_model = None

    def copy(self):
        new_spectrum = Spectrum1D()
        new_spectrum._redshift_model = self._redshift_model.copy()
        new_spectrum._continuum_model = self._continuum_model.copy()

        if self._line_model is not None:
            new_spectrum._line_model = self._line_model.copy()

        return new_spectrum

    def _change_base(self, base):
        new_spectrum = self.copy()

        new_spectrum.__class__ = type('Spectrum1D', (Spectrum1D, type(base)),
                                      base.__dict__)

        return new_spectrum

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

    @property
    def tau(self):
        new_compound = (RedshiftScaleFactor(self.redshift).inverse
                        | self._line_model()
                        | Scale(1. / (1 + self.redshift)))

        return self._change_base(new_compound)

    @property
    def flux(self):
        new_compound = (RedshiftScaleFactor(self.redshift).inverse
                        | (Linear1D(0.0, 1.0)
                           + (self._line_model() | FluxConvert()))
                        | Scale(1. / (1 + self.redshift)))

        return self._change_base(new_compound)

    @property
    def flux_decrement(self):
        new_compound = (RedshiftScaleFactor(self.redshift).inverse
                        | (Linear1D(0.0, 1.0)
                           + (self._line_model() | FluxDecrementConvert()))
                        | Scale(1. / (1 + self.redshift)))

        return self._change_base(new_compound)

    def velocity_space(self, velocity, center):
        if issubclass(type(self), _CompoundModel):
            return (WavelengthConvert(center) | self)(velocity)
        else:
            raise SpectrumModelNotImplemented(
                "No spectral axis specified. Please method chain with `tau`, "
                "`flux`, or `flux_decrement`.")

    def wavelength_space(self, wavelength):
        if issubclass(type(self), _CompoundModel):
            return self(wavelength)
        else:
            raise SpectrumModelNotImplemented(
                "No spectral axis specified. Please method chain with `tau`, "
                "`flux`, or `flux_decrement`.")

    def line_mask(self, x):
        masks = [line.mask(x) for line in self._lines]

        return np.logical_or.reduce(masks)


