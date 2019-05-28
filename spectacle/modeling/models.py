import operator
from functools import wraps
import logging

import astropy.units as u
import numpy as np
from scipy.stats import chisquare
from astropy.convolution import Kernel1D
from astropy.modeling import Fittable1DModel, FittableModel, Parameter
from astropy.modeling.models import Const1D, RedshiftScaleFactor
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.table import QTable
from collections import OrderedDict

from .converters import FluxConvert, FluxDecrementConvert
from .lsfs import COSLSFModel, GaussianLSFModel, LSFModel
from .profiles import OpticalDepth1D
from ..utils.misc import DOPPLER_CONVERT
from ..analysis import delta_v_90, equivalent_width, full_width_half_max
from ..analysis.region_finder import find_regions

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
        elif len(self.lines) == 0:
            return {'x': u.Unit('km/s')}
        else:
            return {'x': None}

    @property
    def input_units_equivalencies(self):
        rest_wavelength = self.rest_wavelength

        if len(self.lines) > 0 and (self.is_single_ion or
                                    self.rest_wavelength.value == 0):
            rest_wavelength = self.lines[0].lambda_0.quantity

        disp_equiv = u.spectral() + DOPPLER_CONVERT[
            self._velocity_convention](rest_wavelength)

        return {'x': disp_equiv}

    def __new__(cls, lines=None, continuum=None, z=None, lsf=None, output=None,
                velocity_convention=None, rest_wavelength=None, copy=False,
                input_redshift=None, **kwargs):
        # If the cls already contains parameter attributes, assume that this is
        # being called as part of a copy operation and return the class as-is.
        if (lines is None and continuum is None and z is None and
                output is None and velocity_convention is None and
                rest_wavelength is None):
            return super().__new__(cls)

        output = output or 'optical_depth'
        velocity_convention = velocity_convention or 'relativistic'
        rest_wavelength = rest_wavelength or u.Quantity(0, 'Angstrom')
        z = z or 0
        input_redshift = input_redshift or 0

        # If no continuum is provided, or the continuum provided is not a
        # model, use a constant model to represent the continuum.
        if continuum is not None:
            if not issubclass(type(continuum), FittableModel):
                if isinstance(continuum, (float, int)):
                    continuum = Const1D(amplitude=continuum, fixed={'amplitude': True})
                else:
                    raise ValueError("Continuum must be a number or `FittableModel`.")
            else:
                # If the continuum model is an astropy model, ensure that it
                # can handle inputs with units or wrap otherwise.
                if not continuum._supports_unit_fitting:
                    continuum = _wrap_unitless_model(continuum)
        else:
            continuum = Const1D(amplitude=0)

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
        rs = RedshiftScaleFactor(z, fixed={'z': True}, name="redshift").inverse
        irs = RedshiftScaleFactor(input_redshift, fixed={'z': True},
                                  name="input_redshift")

        if lines is not None and len(_lines) > 0:
            ln = np.sum(_lines)

            if output == 'flux_decrement':
                compound_model = rs | ((ln | FluxDecrementConvert()) + continuum)
            elif output == 'flux':
                compound_model = rs | ((ln | FluxConvert()) + continuum)
            else:
                compound_model = rs | (ln + continuum)
        else:
            compound_model = rs | continuum

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
        data_units = OrderedDict()

        # Attach all of the compound model parameters to this model
        for param_name in compound_model.param_names:
            param = getattr(compound_model, param_name)
            members[param_name] = param.copy(fixed=param.fixed)
            data_units[param_name] = param.unit

        members['_parameter_units_for_data_units'] = lambda *pufdu: data_units

        new_cls = type('Spectral{}'.format(compound_model.__name__),
                       (cls, ), members)

        # Ensure that the class is recorded in the global scope so that the
        # serialization done can access the stored class definitions.
        new_cls.__module__ = '__main__'
        # globals()[new_cls.__name__] = new_cls
        globals()[compound_model.__name__] = compound_model.__class__

        instance = super().__new__(new_cls)

        # Define the instance-level parameters
        setattr(instance, '_continuum', continuum)
        setattr(instance, '_compound_model', compound_model)
        setattr(instance, '_velocity_convention', velocity_convention)
        setattr(instance, '_rest_wavelength', rest_wavelength)
        setattr(instance, '_output', output)

        return instance

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, x, *args, **kwargs):
        if isinstance(x, u.Quantity):
            if x.unit.physical_type in ('length', 'frequency'):
                for k in self.fixed:
                    if 'delta_v' in k:
                        getattr(self, k).fixed = True
                    elif 'delta_lambda' in k:
                        getattr(self, k).fixed = False
            elif x.unit.physical_type == 'speed':
                for k in self.fixed:
                    if 'delta_v' in k:
                        getattr(self, k).fixed = False
                    elif 'delta_lambda' in k:
                        getattr(self, k).fixed = True

        return super().__call__(x, *args, **kwargs)

    def evaluate(self, x, *args, **kwargs):
        # For the input dispersion to be unit-ful, especially when fitting
        x = u.Quantity(x, 'km/s') \
            if self.is_single_ion or len(self.lines) == 0 \
            else u.Quantity(x, 'Angstrom')

        # For the parameters to be unit-ful especially when used in fitting.
        # TODO: fix arguments being passed with extra dimension.
        args = [u.Quantity(val[0], unit) if unit is not None else val[0]
                for val, unit in zip(args, self._parameter_units_for_data_units().values())]

        return self._compound_model.__class__(*args, **kwargs)(x)

    def rejection_criteria(self, x, y, auto_fit=True):
        """
        Implementation of the Akaike Information Criteria with Correction
        (AICC) (Akaike 1974; Liddle 2007; King et al. 2011). Used to determine
        whether lines can be safely removed from the compound model without
        loss of information.

        Parameters
        ----------
        x : :class:`~astropy.units.Quantity`
            The dispersion data.
        y : array-like
            The expected flux or tau data.
        auto_fit : bool
            Whether the model fit should be re-evaluated for every removed
            line.

        Returns
        -------
        final_model : :class:`~spectacle.Spectral1D`
            The new spectral model with the least complexity.
        """
        base_aicc = self._aicc(x, y, self)
        final_model = self
        finished = False

        while not finished:
            for i in range(len(final_model.lines)):
                lines = [x for x in final_model.lines]
                lines.pop(i)
                new_spec = self._copy(lines=lines)
                aicc = self._aicc(x, y, new_spec)

                # print("Testing", aicc, "<", base_aicc)
                if aicc < base_aicc:
                    # print("Removing line. {} remaining.".format(len(lines)))
                    final_model = new_spec
                    base_aicc = aicc
                    break
            else:
                finished = True

        return final_model

    def _aicc(self, x, y, model):
        chi2, pval = chisquare(f_obs=model(x)**2, f_exp=y**2)
        p = len([k for x in model.lines for k, v in x.fixed.items() if not v])
        n = x.size

        return chi2 + (2 * p * n) / (n - p - 1)

    @property
    def continuum(self):
        return self._continuum

    @property
    def velocity_convention(self):
        return self._velocity_convention

    @property
    def rest_wavelength(self):
        return self._rest_wavelength

    @rest_wavelength.setter
    def rest_wavelength(self, value):
        self._rest_wavelength = value

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
                     if isinstance(x, RedshiftScaleFactor)
                     and x.name == 'redshift')).inverse.z.value

    def with_redshift(self, value):
        """Generate a new spectral model with the given redshift."""
        return self._copy(z=value)

    def _input_redshift(self):
        """The defined redshift at which dispersion values are provided."""
        return next((x for x in self._compound_model
                     if isinstance(x, RedshiftScaleFactor)
                     and x.name == 'input_redshift')).z.value

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
        return self._output

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
            rest_wavelength=self.rest_wavelength,
            # input_redshift=self._input_redshift()
        )

        new_kwargs.update(kwargs)

        return Spectral1D(**new_kwargs, copy=True)

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

    def with_continuum(self, continuum):
        """New spectral model defined with a different continuum."""
        return self._copy(continuum=continuum)

    def with_lsf(self, kernel=None, **kwargs):
        """New spectral model with a line spread function."""
        return self._copy(lsf=kernel, **kwargs)

    def with_line(self, *args, reset=False, **kwargs):
        """
        Add a new line to the spectral model.

        Returns
        -------
        : :class:`~spectacle.modeling.models.Spectral1D`
            The new spectral model.
        """
        args = list(args)

        if len(args) > 0:
            if isinstance(args[0], OpticalDepth1D):
                new_line = args[0]
            elif isinstance(args[0], str):
                name = args.pop(0)
                new_line = OpticalDepth1D(name=name, *args, **kwargs)
            elif isinstance(args[0], u.Quantity):
                lambda_0 = args.pop(0)
                new_line = OpticalDepth1D(lambda_0=lambda_0, *args, **kwargs)
        else:
            new_line = OpticalDepth1D(*args, **kwargs)

        return self._copy(
            lines=self.lines + [new_line] if not reset else [new_line])

    def with_lines(self, lines, reset=False):
        """
        Create a new spectral model with the added lines.

        Parameters
        ----------
        lines : list
            List of :class:`~spectacle.modeling.profiles.OpticalDepth1D` line
            objects.

        Returns
        -------
        : :class:`~spectacle.modeling.models.Spectral1D`
            The new spectral model.
        """
        if not all([isinstance(x, OpticalDepth1D) for x in lines]):
            raise ValueError("All lines must be `OpticalDepth1D` objects.")

        return self._copy(lines=self.lines + lines if not reset else lines)

    @u.quantity_input(x=['length', 'speed', 'frequency'])
    def line_stats(self, x):
        """
        Calculate statistics over individual line profiles.

        Parameters
        ----------
        x : :class:`~u.Quantity`
            The input dispersion in either wavelength/frequency or velocity
            space.

        Returns
        -------
        tab : :class:`~astropy.table.QTable`
            A table detailing the calculated statistics.
        """
        tab = QTable(names=['name', 'wave', 'col_dens', 'v_dop',
                            'delta_v', 'delta_lambda', 'ew', 'dv90', 'fwhm'],
                     dtype=('S10', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                            'f8', 'f8'))

        tab['wave'].unit = u.AA
        tab['v_dop'].unit = u.km / u.s
        tab['ew'].unit = u.AA
        tab['dv90'].unit = u.km / u.s
        tab['fwhm'].unit = u.AA
        tab['delta_v'].unit = u.km / u.s
        tab['delta_lambda'].unit = u.AA

        for line in self.lines:
            disp_equiv = u.spectral() + DOPPLER_CONVERT[
                self.velocity_convention](line.lambda_0.quantity)

            with u.set_enabled_equivalencies(disp_equiv):
                vel = x.to('km/s')
                wav = x.to('Angstrom')

            # Generate the spectrum1d object for this line profile
            ew = equivalent_width(wav, line(wav))
            dv90 = delta_v_90(vel, line(vel))
            fwhm = full_width_half_max(wav, line(wav))

            tab.add_row([line.name,
                         line.lambda_0,
                         line.column_density,
                         line.v_doppler,
                         line.delta_v,
                         line.delta_lambda,
                         ew,
                         dv90,
                         fwhm])

        return tab

    @u.quantity_input(x=['length', 'speed', 'frequency'], rest_wavelength='length')
    def region_stats(self, x, rest_wavelength, rel_tol=1e-2, abs_tol=1e-5):
        """
        Calculate statistics over arbitrary line regions given some tolerance
        from the continuum.

        Parameters
        ----------
        x : :class:`~u.Quantity`
            The input dispersion in either wavelength/frequency or velocity
            space.
        rest_wavelength : :class:`~u.Quantity`
            The rest frame wavelength used in conversions between wavelength/
            frequency and velocity space.
        rel_tol : float
            The relative tolerance parameter.
        abs_tol : float
            The absolute tolerance parameter.

        Returns
        -------
        tab : :class:`~astropy.table.QTable`
            A table detailing the calculated statistics.
        """
        y = self(x)

        if self.output_type == 'flux':
            y = self.continuum(x) - y
        else:
            y -= self.continuum(x)

        # Calculate the regions in the raw data
        # absolute(a - b) <= (atol + rtol * absolute(b))
        regions = {(reg[0], reg[1]): []
                   for reg in find_regions(y, rel_tol=rel_tol, abs_tol=abs_tol)}
        tab = QTable(names=['region_start', 'region_end', 'rest_wavelength',
                            'ew', 'dv90', 'fwhm'],
                     dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8'))

        tab['region_start'].unit = x.unit
        tab['region_end'].unit = x.unit
        tab['rest_wavelength'].unit = u.AA
        tab['ew'].unit = u.AA
        tab['dv90'].unit = u.km / u.s
        tab['fwhm'].unit = u.AA

        for mn_bnd, mx_bnd in regions:
            mask = (x > x[mn_bnd]) & (x < x[mx_bnd])
            x_reg = x[mask]
            y_reg = y[mask]

            disp_equiv = u.spectral() + DOPPLER_CONVERT[
                self.velocity_convention](rest_wavelength)

            with u.set_enabled_equivalencies(disp_equiv):
                vel = x_reg.to('km/s')
                wav = x_reg.to('Angstrom')

            # Generate the spectrum1d object for this line profile
            ew = equivalent_width(wav, y_reg)
            dv90 = delta_v_90(vel, y_reg)
            fwhm = full_width_half_max(wav, y_reg)

            tab.add_row([x[mn_bnd],
                         x[mx_bnd],
                         rest_wavelength,
                         ew,
                         dv90,
                         fwhm])

        return tab


def _wrap_unitless_model(model):
    """
    Wraps a model that does not support inputs with units, decorating its
    evaluate method to strip any input units.

    Parameters
    ----------
    model : :class:`astropy.modeling.models.Fittable1DModel`
        The model whose evaluate method will be wrapped.

    Returns
    -------
    model : :class:`astropy.modeling.models.Fittable1DModel`
        The model instance whose evaluate method has been wrapped.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(x, *args, **kwargs):
            if isinstance(x, u.Quantity):
                x = x.value

            return func(x, *args, **kwargs)
        return wrapper

    setattr(model, 'evaluate', decorator(model.evaluate))

    return model


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