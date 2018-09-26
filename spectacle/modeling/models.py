import numpy as np
import astropy.units as u
from astropy.modeling.models import Const1D, RedshiftScaleFactor, Scale
from astropy.modeling import Fittable1DModel
import operator

from ..registries.lines import line_registry

__all__ = ['Spectral1D']


class DynamicFittable1DModelMeta(type):
    def __call__(cls, lines, continuum=None, z=0, *args, **kwargs):
        if continuum is None or not isinstance(continuum, Fittable1DModel):
            continuum = Const1D(amplitude=0)

        _lines = []

        if isinstance(lines, Fittable1DModel):
            _lines.append(lines)
        elif isinstance(lines, list):
            for line in lines:
                if isinstance(line, str):
                    _lines.append(line_registry.with_name(line))
                elif isinstance(line, u.Quantity):
                    _lines.append(line_registry.with_lambda(line))
                elif isinstance(line, Fittable1DModel):
                    _lines.append(line)

        compound_model = (RedshiftScaleFactor(z, fixed={'z': True}).inverse |
                          (continuum + np.sum(lines)).__class__ |
                          Scale(1. / (1 + z), fixed={'factor': True}))

        class Spectral1D(compound_model):
            inputs = ('x',)
            outputs = ('y',)
            input_units_allow_dimensionless = True
            input_units = {'x': u.Unit('km/s')}

            @property
            def input_units_equivalencies(self):
                return {'x': u.spectral() +
                             u.doppler_relativistic(
                                 self._rest_wavelength)}

            def __init__(self, rest_wavelength=None, *args, **kwargs):
                if rest_wavelength is not None:
                    self._rest_wavelength = rest_wavelength

                super().__init__(*args, **kwargs)

            def __call__(self, x, *args, **kwargs):
                dispersion_equivalencies = u.spectral() + u.doppler_relativistic(
                    self._rest_wavelength)

                unitless_cls = _strip_units(self)
                unitless_cls.input_units_allow_dimensionless = {'x': True}
                unitless_cls.input_units = {'x': u.Unit('km/s')}
                unitless_cls.input_units_equivalencies = {'x': dispersion_equivalencies}

                if isinstance(x, u.Quantity):
                    with u.set_enabled_equivalencies(dispersion_equivalencies):
                        x = x.to('km/s').value

                return unitless_cls().__call__(x, *args, **kwargs)

        return type.__call__(Spectral1D, *args, **kwargs)


class Spectral1D(metaclass=DynamicFittable1DModelMeta):
    pass


def _strip_units(compound_model, x=None):
    operators = {'+': operator.add,
                 '-': operator.sub,
                 '*': operator.mul,
                 '/': operator.truediv,
                 '**': operator.pow,
                 '&': operator.and_,
                 '|': operator.or_}

    leaf_idx = -1

    def getter(idx, model):
        # By indexing on self[] this will return an instance of the
        # model, with all the appropriate parameters set, which is
        # currently required to return an inverse
        sub_mod = compound_model[idx]

        for pn in sub_mod.param_names:
            param = getattr(sub_mod, pn)

            if param.unit is not None:
                if x is not None and isinstance(x, u.Quantity):
                    with u.set_enabled_equivalencies(
                            u.spectral() + u.doppler_relativistic(
                                compound_model._rest_wavelength)):
                        quant = param.quantity.to(x.unit)
                else:
                    quant = u.Quantity(param.value)

                param.value = quant.value
                param._set_unit(None, force=True)

        return sub_mod

    return compound_model._tree.evaluate(operators, getter=getter).__class__