import numpy as np
import astropy.units as u
from astropy.modeling.models import Const1D, RedshiftScaleFactor, Scale
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.fitting import _FitterMeta, LevMarLSQFitter
import operator
import logging

from ..registries.lines import line_registry
from .profiles import OpticalDepth1D

__all__ = ['Spectral1D']


class DynamicFittable1DModelMeta(type):
    """
    Meta class that acts as a factory for creating compound models with
    variable collections of sub models. This can take a collection of
    :class:`~spectacle.modeling.profiles.OpticalDepth1D` and return a new
    compound model.
    """
    def __call__(cls, lines, continuum=None, z=0, rest_wavelength=0 * u.AA, *args,
                 **kwargs):
        # If no continuum is provided, or the continuum provided is not a
        # model, use a constant model to represent the continuum.
        if continuum is None or not isinstance(continuum, Fittable1DModel):
            continuum = Const1D(amplitude=0, fixed={'amplitude': True})

        # Parse the lines argument which can be a list, a quantity, or a string.
        _lines = []

        if isinstance(lines, Fittable1DModel):
            _lines.append(lines)
        elif isinstance(lines, str):
            _lines.append(OpticalDepth1D(name=lines))
        elif isinstance(lines, list):
            for line in lines:
                if isinstance(line, str):
                    _lines.append(OpticalDepth1D(name=line))
                elif isinstance(line, u.Quantity):
                    _lines.append(OpticalDepth1D(lambda_0=line))
                elif isinstance(line, Fittable1DModel):
                    _lines.append(line)

        # Compose the line-based compound model taking into consideration
        # the redshift, continuum, and dispersion conversions.
        compound_model = (DispersionConvert(u.Quantity(rest_wavelength, u.AA)) |
                          RedshiftScaleFactor(z, fixed={'z': True}) |
                          (continuum + np.sum(_lines)).__class__ |
                          Scale(1. / (1 + z), fixed={'factor': True}))

        # Add the definitions inside the original Spectral1D class to the new
        # compound class. Creating a new class seems to be less error-prone
        # than dynamically changing the base of a pre-existing class.
        _cls_kwargs = {}
        _cls_kwargs.update(cls.__dict__)
        Spectral1D = type("Spectral1D", (compound_model,), _cls_kwargs)
        spec1d = Spectral1D()

        # Updating the dict directly on an instance instead of during the type
        # creation above avoids issues of python complaining that the __dict__
        # object does not belong to the class.
        # spec1d.__dict__.update(cls.__dict__)

        return type.__call__(spec1d.__class__, *args, **kwargs)


class Spectral1D(metaclass=DynamicFittable1DModelMeta):
    """
    Base representation of a compound model containing a variable number of
    :class:`~spectacle.modeling.profiles.OpticalDepth1D` line model features.

    Parameters
    ----------
    lines : str, :class:`~OpticalDepth1D`, list
        The line information used to compose the spectral model. This can be
        either a string, in which case the line information is retrieve from the
        ion registry; an instance of :class:`~OpticalDepth1D`; or a list of
        either of the previous two types.
    continuum : :class:`~Fittable1DModel`, optional
        An astropy model representing the continuum of the spectrum. If not
        provided, a :class:`~Const1D` model is used.
    z : float, optional
        The redshift applied to the spectral model. Default = 0.
    rest_wavelength : :class:`~astropy.units.Quantity`, optional
        The rest wavelength used in dispersions conversions between wavelength
        and velocity. Default = 0 Angstroms.
    """
    inputs = ('x',)
    outputs = ('y',)
    input_units_allow_dimensionless = True
    input_units = {'x': u.Unit('km/s')}

    def fit_to(self, x, y, fitter=LevMarLSQFitter()):
        if not fitter.__class__ in _FitterMeta.registry:
            raise Exception("Fitter must be an astropy fitter subclass.")

        # The internal models assume all inputs are in km/s, if provided a
        # quantity object, ensure that it is converted to the proper units.
        if isinstance(x, u.Quantity):
            x = x.to('km/s').value

        # Create a new compound without units that can be used with the
        # astropy fitters, since compound models with units are not
        # currently supported.
        unitless_cls, parameter_units = _strip_units(self)
        fitted_model = fitter(unitless_cls(), x, y)

        # Now we put back the units on the model
        model_with_units = _apply_units(fitted_model, parameter_units)

        return model_with_units()


class DispersionConvert(Fittable1DModel):
    """
    Convert dispersions into velocity space for use internally.

    Arguments
    ---------
    rest_wavelength : :class:`~astropy.units.Parameter`
        Wavelength for use in the equivalency conversions.
    """
    inputs = ('x',)
    outputs = ('x',)

    rest_wavelength = Parameter(default=0, unit=u.AA, fixed=True)

    input_units_allow_dimensionless = {'x': True}
    input_units = {'x': u.Unit('km/s')}

    linear = True
    fittable = True

    @property
    def input_units_equivalencies(self):
        return {'x': u.spectral() + u.doppler_relativistic(
            self.rest_wavelength.value * u.AA)}

    @staticmethod
    def evaluate(x, rest_wavelength):
        """One dimensional Scale model function"""
        disp_equiv = u.spectral() + u.doppler_relativistic(
            u.Quantity(rest_wavelength, u.AA))

        with u.set_enabled_equivalencies(disp_equiv):
            x = u.Quantity(x, u.Unit("km/s"))

        return x.value


# Model arithmetic operators
OPERATORS = {'+': operator.add,
             '-': operator.sub,
             '*': operator.mul,
             '/': operator.truediv,
             '**': operator.pow,
             '&': operator.and_,
             '|': operator.or_}


def _strip_units(compound_model, x=None):
    """
    Remove the units of a given compound model.

    Parameters
    ----------
    compound_model : :class:`~Fittable1D`
        Compound model for which the units will be removed.
    x : :class:`~astropy.units.Quantity`, optional
        The dispersion array that will be passed into the compound model. If
        provided, relevant parameters of the compound model will be converted
        to the unit.

    Returns
    -------
    : :class:`~Fittable1D`
        Compound model without units.
    """
    leaf_idx = -1

    parameter_units = {pn: getattr(sm, pn).unit
                       for sm in compound_model for pn in sm.param_names}

    def getter(idx, model):
        # By indexing on self[] this will return an instance of the
        # model, with all the appropriate parameters set
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

    unitless_model = compound_model._tree.evaluate(OPERATORS, getter=getter).__class__
    unitless_model._parameter_units = parameter_units

    return unitless_model, parameter_units


def _apply_units(compound_model, parameter_units):
    """
    Applies a set of units to a compound model.

    Parameters
    ----------
    compound_model : :class:`~Fittable1D`
        Unitless compound model.
    parameter_units : dict
        Dictionary containing a mapping between parameter names in the sub
        models of the compound model and the units to be associated with them.

    Returns
    -------
    : :class:`~Fittable1D`
        Compound model with units.
    """
    leaf_idx = -1

    def getter(idx, model):
        # By indexing on self[] this will return an instance of the
        # model, with all the appropriate parameters set
        sub_mod = compound_model[idx]

        for pn in sub_mod.param_names:
            param = getattr(sub_mod, pn)

            unit = parameter_units.get(pn)
            param._set_unit(unit, force=True)

        return sub_mod

    return compound_model._tree.evaluate(OPERATORS, getter=getter).__class__