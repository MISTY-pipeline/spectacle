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
                     FluxConvert, FluxDecrementConvert, SmartScale, Masker)


class SpectrumModelNotImplemented(Exception): pass

class IncompleteLineInformation(Exception): pass


class SpectrumModel(DispersionConvert | RedshiftScaleFactor | SmartScale):
    def __init__(self, center=0 * u.Angstrom, z=0):
        self._center = u.Quantity(center, 'Angstrom')

        self._redshift_model = RedshiftScaleFactor(z).inverse
        self._continuum_model = Linear1D(0 * u.Unit('1/Angstrom'), 1)
        self._line_model = None

        self._compound_model = None

    # def __call__(self, x, mask=None, *args, **kwargs):
    #     return self._compound_model(x, *args, **kwargs)

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
        new_spec_mod = SpectrumModel()


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
                 delta_lambda=None, name=None, line_list=None, tied=None,
                 fixed=None):
        tied = tied or {}
        fixed = fixed or {}

        if line_list is not None:
            line_list = [line_registry.with_name(x) for x in line_list]
            line_list = line_registry[line_registry['name'] == line_list]
        else:
            line_list = line_registry

        if name is not None:
            line = line_list.with_name(name)
            lambda_0 = line['wave'] * u.Angstrom
        elif lambda_0 is not None:
            ind = find_nearest(line_list['wave'], lambda_0)
            line = line_list[ind]
            name = line['name']
        else:
            raise IncompleteLineInformation(
                "Not enough information to construction absorption line "
                "profile. Please provide at least a name or centroid.")

        def _tie_nearest(mod, col_name):
            ind = find_nearest(line_list['wave'], mod.lambda_0)
            line = line_list[ind]

            return line[col_name]

        if gamma is None:
            gamma = line['gamma']

            tied.update({
                'gamma': lambda cmod, col_name='gamma':
                    _tie_nearest(cmod, col_name)})

        if f_value is None:
            f_value = line['osc_str']

            tied.update({
                'f_value': lambda cmod, col_name='osc_str':
                    _tie_nearest(cmod, col_name)})

        tau_prof = TauProfile(lambda_0=lambda_0, column_density=column_density,
                              v_doppler=v_doppler, gamma=gamma, f_value=f_value,
                              delta_v=delta_v, delta_lambda=delta_lambda,
                              name=name, tied=tied, fixed=fixed)

        self._line_model = tau_prof if self._line_model is None else self._line_model + tau_prof

        return self

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


