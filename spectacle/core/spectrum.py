import logging

import astropy.units as u
from astropy.modeling import models
from astropy.modeling.models import Linear1D

from ..modeling.converters import (DispersionConvert, FluxConvert,
                                   FluxDecrementConvert)
from ..modeling.custom import Redshift, SmartScale
from ..modeling.profiles import TauProfile


class SpectrumModelNotImplemented(Exception):
    pass


class InappropriateModel(Exception):
    pass


class NoLines(Exception):
    pass


class Spectrum1D:
    def __init__(self, center=0, redshift=None, continuum=None):
        self._center = u.Quantity(center, 'Angstrom')

        self._redshift_model = Redshift(
            **(redshift or {'z': 0})).inverse
        self._continuum_model = Linear1D(
            **(continuum or {'slope': 0 * u.Unit('1/Angstrom'),
                             'intercept': 1 * u.Unit('')}))
        self._line_model = None

        self._lsf = None

    def copy(self):
        new_spectrum = Spectrum1D(center=self.center.value * self.center.unit)
        new_spectrum._redshift_model = self._redshift_model.copy()
        new_spectrum._continuum_model = self._continuum_model.copy()

        if self._line_model is not None:
            new_spectrum._line_model = self._line_model.copy()

        return new_spectrum

    @property
    def center(self):
        """
        The central wavelength value.
        """
        return self._center

    @center.setter
    def center(self, value):
        """
        Define the center wavelength for this spectrum model. The center
        dictates wavelength to velocity space dispersion conversions.

        Parameters
        ----------
        value : `~astropy.units.Quantity`
            Quantity object containing the center value and a unit of type
            *length*.
        """
        self._center = value

    @property
    def redshift(self):
        """
        Read the current redshift model.

        Returns
        -------
        : `~astropy.modeling.modeling.Fittable1DModel`
            The redshift model in the spectrum compound model.
        """

        return self._redshift_model

    @redshift.setter
    def redshift(self, z):
        """
        Set the redshift value to use in the compound spectrum model.

        Parameters
        ----------
        z : float
            The redshift value to use.
        """
        # TODO: include check on the input arguments
        self._redshift_model = Redshift(z=z).inverse

    @property
    def continuum(self):
        """
        Read the current continuum model.

        Returns
        -------
        : `~astropy.modeling.modeling.Fittable1DModel`
            The continuum model in the spectrum compound model.
        """
        return self._continuum_model

    @continuum.setter
    def continuum(self, model='Linear1D', *args, **kwargs):
        """
        Set the continuum model used in the spectrum compound model to one of
        a user-defined model from within the astropy modeling package.

        Parameters
        ----------
        model : str
            The class name of the model as a string. This model must exist in
            the `~astropy.modeling.modeling` package.
        args :
        kwargs

        Returns
        -------

        """
        if not hasattr(models, model):
            logging.error(
                "No available model named %s. Modle must be one of \n%s.",
                model, [cls.__name__ for cls in
                        vars()['FittableModel1D'].__subclasses()])

            return

        self._continuum_model = getattr(models, model)(*args, **kwargs)

    def add_line(self, *args, **kwargs):
        """
        Create an absorption line Voigt profile model and add it to the
        compound spectral model.

        Parameters
        ----------
        See `~spectacle.modeling.profiles.TauProfile` for method arguments.

        Returns
        -------
        : `~spectacle.core.spectrum.Spectrum1D`
            The spectrum compound model including the new line.
        """
        tau_prof = TauProfile(*args, **kwargs)

        self._line_model = tau_prof if self._line_model is None \
            else self._line_model + tau_prof

        return self

    @property
    def lsf(self):
        """
        Reads the current LSF model used in the compound spectrum model.

        Returns
        -------
        : `~astropy.modeling.modeling.Fittable1D`
            LSF kernel model.
        """
        return self._lsf

    @lsf.setter
    def lsf(self, value):
        """
        Sets the LSF model to be used in the compound spectrum model.

        Parameters
        ----------
        value : `~astropy.modeling.modeling.Fittable1D`
            The model to use. Must take 'y' as input and give 'y' as output.
        """
        self._lsf = value

    def _rebuild_compound_model(self, comp_mod):
        comp_mod.input_units = comp_mod[0].input_units
        comp_mod.input_units_equivalencies = comp_mod[0].input_units_equivalencies

    @property
    def optical_depth(self):
        """
        Compound spectrum model in tau space.
        """
        dc = DispersionConvert(self._center)
        rs = self._redshift_model
        ss = SmartScale(tied={
            'factor': lambda mod: 1. / (1 + self._redshift_model.z)
        })
        lm = self._line_model

        comp_mod = (dc | rs | lm | ss) if lm is not None else (dc | rs | ss)

        return type('OpticalDepth', (comp_mod.__class__,),
                    {'inputs': ('x',),
                     'outputs': ('y',),
                     'input_units_script': True,
                     'inputs_units': input_units={'x': u.Unit('Angstrom')}})

    @property
    def flux(self):
        """
        Compound spectrum model in flux space.
        """
        dc = DispersionConvert(self._center)
        rs = self._redshift_model
        ss = SmartScale(tied={
            'factor': lambda mod: 1. / (1 + self._redshift_model.z)
        })
        cm = self._continuum_model
        lm = self._line_model
        fc = FluxConvert()

        comp_mod = (dc | rs | (cm + (lm | fc)) |
                    ss) if lm is not None else (dc | rs | cm | ss)

        return comp_mod.rename("FluxModel")

    @property
    def flux_decrement(self):
        """
        Compound spectrum model in flux decrement space.
        """
        dc = DispersionConvert(self._center)
        rs = self._redshift_model
        cm = self._continuum_model
        lm = self._line_model
        fd = FluxDecrementConvert()
        ss = SmartScale(tied={
            'factor': lambda mod: 1. / (1 + self._redshift_model.z)
        })

        comp_mod = (dc | rs | (cm + (lm | fd)) |
                    ss) if lm is not None else (dc | rs | cm | ss)

        return comp_mod.rename("FluxDecrementModel")


def model_factory(bases, name="BaseModel"):
    class BaseModel(Fittable1D):
        inputs = ('x',)
        outputs = ('y',)

        input_units_strict = True

        input_units = {'x': u.Unit('Angstrom')}

    new_class = type(name, bases, )
