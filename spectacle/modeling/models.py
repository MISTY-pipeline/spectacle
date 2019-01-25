import logging
import operator

import astropy.units as u
import numpy as np
from astropy.convolution import Kernel1D
from astropy.modeling import Fittable1DModel, FittableModel
from astropy.modeling.core import _CompoundModelMeta
from astropy.modeling.models import Const1D, RedshiftScaleFactor

from .converters import FluxConvert, FluxDecrementConvert
from .lsfs import COSLSFModel, GaussianLSFModel, LSFModel
from .profiles import OpticalDepth1D
from ..fitting.curve_fitter import CurveFitter
from ..utils.misc import DOPPLER_CONVERT
from astropy.modeling import Parameter
from astropy.units.equivalencies import doppler_optical

__all__ = ['Spectral1D']


class Spectral1D(Fittable1DModel):
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
    lsf : :class:`~spectacle.modeling.lsfs.LSFModel`, :class:`~astropy.convolution.Kernel1D`, str, optional
        The line spread function applied to the spectral model. It can be a
        pre-defined kernel model, or a convolution kernel, or a string
        referencing the built-in Hubble COS lsf, or a Gaussian lsf. Optional
        keyword arguments can be passed through.
    """
    inputs = ('x',)
    outputs = ('y',)

    @property
    def input_units_allow_dimensionless(self):
        return {'x': False}

    @property
    def input_units(self):
        if any([x.unit is not None for x in [getattr(self, y) for y in self.param_names]]):
            if self.is_single_ion:
                return {'x': u.Unit('km/s')}
            else:
                return {'x': u.Unit('Angstrom')}
        else:
            return {'x': None}

    @property
    def input_units_equivalencies(self):
        rest_wavelength = self.lines[0].lambda_0.quantity \
            if self.is_single_ion else self.rest_wavelength

        disp_equiv = u.spectral() + DOPPLER_CONVERT[
            self._velocity_convention](rest_wavelength)

        return {'x': disp_equiv}

    def __new__(cls, lines=None, continuum=None, z=None, lsf=None, output=None,
                velocity_convention=None, rest_wavelength=None, *args, **kwargs):
        # If the cls already contains parameter attributes, assume that this is
        # being called as part of a copy operation and return the class as-is.
        if (lines is None and continuum is None and z is None and
                output is None and velocity_convention is None and
                rest_wavelength is None):
            return super().__new__(cls)

        output = output or 'flux'
        velocity_convention = velocity_convention or 'relativistic'
        rest_wavelength = rest_wavelength or u.Quantity(0, 'Angstrom')

        # If no continuum is provided, or the continuum provided is not a
        # model, use a constant model to represent the continuum.
        if continuum is not None:
            if not issubclass(type(continuum), FittableModel):
                if isinstance(continuum, (float, int)):
                    continuum = Const1D(amplitude=continuum, fixed={'amplitude': True})
                else:
                    raise ValueError("Continuum must be a number or `FittableModel`.")
        else:
            continuum = Const1D(amplitude=0, fixed={'amplitude': True})

        if output not in ('flux', 'flux_decrement', 'optical_depth'):
            raise ValueError("Parameter 'output' must be one of 'flux', "
                             "'flux_decrement', 'optical_depth'.")

        # Parse the lines argument which can be a list, a quantity, or a string
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

        # Parse the lsf information, if provided
        if lsf is not None and not isinstance(lsf, LSFModel):
            if isinstance(lsf, Kernel1D):
                lsf = LSFModel(kernel=lsf)
            elif isinstance(lsf, str):
                if lsf == 'cos':
                    lsf = COSLSFModel()
                elif lsf == 'gaussian':
                    lsf = GaussianLSFModel(kwargs.pop('stddev', 1))
            else:
                raise ValueError("Kernel must be of type 'LSFModel', or "
                                 "'Kernel1D'; or a string with value 'cos' "
                                 "or 'gaussian'.")

        # Compose the line-based compound model taking into consideration
        # the redshift, continuum, and dispersion conversions.
        rs = RedshiftScaleFactor(z, fixed={'z': True}).inverse

        if lines is not None:
            ln = np.sum(_lines)

            if output == 'flux_decrement':
                compound_model = ((rs | (continuum + ln | FluxDecrementConvert())) | rs.inverse)
            elif output == 'flux':
                compound_model = ((rs | continuum + (ln | FluxConvert())) | rs.inverse)
            else:
                compound_model = (rs | (continuum + ln) | rs.inverse)
        else:
            compound_model = (rs | continuum | rs.inverse)

        # Check for any lsf kernels that have been added
        if lsf is not None:
            compound_model |= lsf

        # Model parameter members are setup in the model's compound meta class.
        # After we've attached the parameters to this fittable model, call the
        # __new__ and __init__ meta methods again to ensure proper creation.
        members = {}
        members.update(cls.__dict__)

        # Delete all previous parameter definitions living on the class
        for k, v in members.items():
            if isinstance(v, Parameter):
                delattr(cls, k)

        # Create a dictionary to pass as the parameter unit definitions to the
        # new class. This ensures fitters know this model supports units.
        data_units = {}

        # Attach all of the compound model parameters to this model
        for param_name in compound_model.param_names:
            param = getattr(compound_model, param_name)
            members[param_name] = param.copy()
            data_units[param_name] = param.unit

        # setattr(cls, '_parameter_units_for_data_units',
        #         lambda *args: data_units)
        members['_parameter_units_for_data_units'] = lambda *args: data_units

        # Since the fitting machinery makes a copy of the model object, attach
        # what would be instance-level attributes to the class.
        # setattr(cls, '_continuum', continuum)
        # setattr(cls, '_compound_model', compound_model)
        # setattr(cls, '_velocity_convention', velocity_convention)
        # setattr(cls, '_rest_wavelength', rest_wavelength)

        cls = type('Spectral1D', (cls, ), members)
        instance = super().__new__(cls)

        # Define the instance-level parameters
        setattr(instance, '_continuum', continuum)
        setattr(instance, '_compound_model', compound_model)
        setattr(instance, '_velocity_convention', velocity_convention)
        setattr(instance, '_rest_wavelength', rest_wavelength)

        return instance

    def __init__(self, *args, **kwargs):
        super().__init__()

    def evaluate(self, x, *args, **kwargs):
        # For the input dispersion to be unit-ful, especially when fitting
        x = u.Quantity(x, 'km/s') if self.is_single_ion else u.Quantity(x, 'Angstrom')

        # For the parameters to be unit-ful especially when used in fitting
        args = [u.Quantity(val, unit) if unit is not None else val
                for val, unit in zip(args, self._parameter_units_for_data_units().values())]

        return self._compound_model.__class__(*args, **kwargs)(x)

    @property
    def continuum(self):
        return self._continuum

    @property
    def velocity_convention(self):
        return self._velocity_convention

    @property
    def rest_wavelength(self):
        return self._rest_wavelength

    @property
    def redshift(self):
        """
        The redshift at which the data given to.

        Returns
        -------
        : float
            The redshift value.
        """
        return next((x for x in self._compound_model
                     if isinstance(x, RedshiftScaleFactor))).inverse.z.value

    @property
    def lines(self):
        """
        The collection of profiles representing the absorption or emission
        lines in the spectrum model.

        Returns
        -------
        : list
            A list of :class:`~spectacle.modeling.profiles.Voigt1D` models.
        """
        return [x for x in self._compound_model if isinstance(x, OpticalDepth1D)]

    @property
    def is_single_ion(self):
        """
        Whether this spectrum represents a collection of single ions or a
        collection of multiple different ions.

        Returns
        -------
        : bool
            Is the spectrum composed of a collection of single ions.
        """
        return len(set([x.name for x in self.lines])) == 1

    @property
    def lsf_kernel(self):
        return next((x for x in self._compound_model if isinstance(x, LSFModel)), None)

    @property
    def output_type(self):
        """
        The data output of this spectral model. It could one of 'flux',
        'flux_decrement', or 'optical_depth'.

        Returns
        -------
        : str
            The output type of the model.
        """
        for x in self._compound_model:
            if isinstance(x, FluxConvert):
                return 'flux'
            elif isinstance(x, FluxDecrementConvert):
                return 'flux_decrement'
        else:
            return 'optical_depth'

    def _copy(self, **kwargs):
        """
        Copy the spectral model, optionally overriding any previous values.

        Parameters
        ----------
        kwargs : dict
            A dictionary holding the values desired to be overwritten.

        Returns
        -------
        : :class:`~spectacle.modeling.models.Spectral1D`
            The new spectral model.
        """
        new_kwargs = dict(
            lines=self.lines,
            continuum=self.continuum,
            z=self.redshift,
            output=self.output_type,
            lsf=self.lsf_kernel,
            velocity_convention=self.velocity_convention,
            rest_wavelength=self.rest_wavelength)

        new_kwargs.update(kwargs)

        return Spectral1D(**new_kwargs)

    @property
    def as_flux(self):
        """New spectral model that produces flux output."""
        return self._copy(output='flux')

    @property
    def as_flux_decrement(self):
        """New spectral model that produces flux decrement output."""
        return self._copy(output='flux_decrement')

    @property
    def as_optical_depth(self):
        """New spectral model that produces optical depth output."""
        return self._copy(output='optical_depth')

    def with_lsf(self, kernel=None, **kwargs):
        """New spectral model with a line spread function."""
        return self._copy(lsf=kernel, **kwargs)

    def with_line(self, *args, **kwargs):
        """
        Add a new line to the spectral model.

        Returns
        -------
        : :class:`~spectacle.modeling.models.Spectral1D`
            The new spectral model.
        """
        if isinstance(args[0], OpticalDepth1D):
            new_line = args[0]
        else:
            new_line = OpticalDepth1D(*args, **kwargs)

        return self._copy(lines=self.lines + [new_line])


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
    compound_model = compound_model.copy()

    if x is not None:
        compound_model._input_units = {'x': x.unit}

    parameter_units = {pn: getattr(sm, pn).unit
                       for sm in compound_model for pn in sm.param_names}

    def getter(idx, model):
        # By indexing on self[] this will return an instance of the
        # model, with all the appropriate parameters set
        sub_mod = compound_model[idx]

        for pn in sub_mod.param_names:
            param = getattr(sub_mod, pn)

            if param.unit is not None:
                # if x is not None and isinstance(x, u.Quantity):
                #     print("Value", sub_mod.lambda_0.quantity)
                #     with u.set_enabled_equivalencies(
                #             u.spectral() + u.doppler_relativistic(lambda_0)):
                #         quant = param.quantity.to(x.unit)
                # else:
                quant = param.quantity

                param.value = quant.value
                param._set_unit(None, force=True)

        return sub_mod

    unitless_model = compound_model._tree.evaluate(OPERATORS, getter=getter).__class__
    unitless_model._parameter_units = parameter_units

    # Set custom call to return a Spectrum1D object
    # _set_custom_call(unitless_model)

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

    unitful_model = compound_model._tree.evaluate(OPERATORS, getter=getter).__class__

    # Set custom call to return a Spectrum1D object
    # _set_custom_call(unitful_model)

    return unitful_model