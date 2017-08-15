import numpy as np

from astropy.modeling.models import RedshiftScaleFactor, Linear1D, Scale
from astropy.modeling.core import _CompoundModel

from .models import (TauProfile, WavelengthConvert, VelocityConvert,
                     FluxConvert, FluxDecrementConvert)


class SpectrumModelNotImplemented(Exception): pass

class IncompleteLineInformation(Exception): pass


class Spectrum1D:
    def __init__(self, *args, **kwargs):
        self._redshift = 0
        self._continuum = None
        self._lines = []

    def _line_model(self):
        return np.sum(self._lines)

    def _change_base(self, base):
        self.__class__ = type('Spectrum1D', (Spectrum1D, type(base)),
                              base.__dict__)

        return self

    @property
    def redshift(self):
        return self._redshift

    @redshift.setter
    def redshift(self, value):
        self._redshift = max(0, value)

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

        self._lines.append(tau_prof)

        return self

    def add_lines(self, lines):
        self._lines += lines

        return self

