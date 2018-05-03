import logging
from collections import OrderedDict

import astropy.units as u
from astropy.modeling import Fittable1DModel, models
from astropy.table import Table

from ..analysis.resample import Resample
from ..io.registries import line_registry
from ..modeling.converters import (DispersionConvert, FluxConvert,
                                   FluxDecrementConvert)
from ..modeling.custom import Redshift, SmartScale, Linear
from ..modeling.profiles import TauProfile
from ..utils import wave_to_vel_equiv


class SpectrumModelNotImplemented(Exception):
    pass


class InappropriateModel(Exception):
    pass


class NoLines(Exception):
    pass


class Spectrum1D:
    def __init__(self, center=None, ion=None, redshift=None, continuum=None):
        self._center = u.Quantity(center or 0, 'Angstrom')

        if ion is not None:
            ion = line_registry.with_name(ion)
            self._center = ion['wave'] * line_registry['wave'].unit

        self._redshift_model = Redshift(**{'z': redshift or 0})

        if continuum is not None and isinstance(continuum, Fittable1DModel):
            self._continuum_model = continuum
        else:
            self._continuum_model = Linear(
                slope=0 * u.Unit('1/Angstrom'),
                intercept=1 * u.Unit(""),
                fixed={'slope': True, 'intercept': True})

            logging.debug("Default continuum set to a Linear model.")

        self._regions = {}

        self._line_model = None
        self._lsf_model = None
        self._noise_model = None
        self._resample_model = None

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
        return self._redshift_model.z

    @redshift.setter
    def redshift(self, value):
        """
        Set the redshift value to use in the compound spectrum model.

        Parameters
        ----------
        value : float
            The redshift value to use.
        """
        # TODO: include check on the input arguments
        self._redshift_model = Redshift(z=value)

    @property
    def continuum(self):
        """
        Read the current continuum model.

        Returns
        -------
        : `~astropy.modeling.modeling.Fittable1DModel`
            The continuum model in the spectrum compound model.
        """
        dc = DispersionConvert(self._center)
        rs = self._redshift_model.inverse
        ss = SmartScale(
            1. / (1 + self._redshift_model.z),
            fixed={'factor': True})
        cm = self._continuum_model

        return (rs | cm | ss).rename("Continuum Model")

    @continuum.setter
    def continuum(self, value):
        """
        Set the continuum model used in the spectrum compound model to one of
        a user-defined model from within the astropy modeling package.
        """
        if not issubclass(value.__class__, Fittable1DModel):
            raise ValueError("Continuum must inherit from 'Fittable1DModel'.")

        self._continuum_model = value

    @property
    def regions(self):
        """
        Identified absorption regions with references to individual line
        profiles.
        """
        return self._regions

    @regions.setter
    def regions(self, value):
        """
        Identified absorption regions with references to individual line
        profiles.
        """
        self._regions = value

    @property
    def lines(self):
        tab = Table(names=['name'] + list(TauProfile.param_names),
                    dtype=['S10'] + ['f8'] * len(TauProfile.param_names))

        for l in self.line_models:
            tab.add_row([l.name] + list(l.parameters))

        params = [getattr(TauProfile, n) for n in TauProfile.param_names]

        for i, n in enumerate(TauProfile.param_names):
            tab[n].unit = params[i].unit

        return tab

    @property
    def line_model(self):
        return self._line_model

    @property
    def line_models(self):
        if self._line_model is None:
            return []
        elif self.line_model.n_submodels() > 1:
            return [x for x in self.line_model]

        return [self.line_model]

    @property
    def n_components(self):
        """Return the number of identified lines in this spectrum."""
        return len(self.line_model)

    def add_line(self, name=None, model=None, *args, **kwargs):
        """
        Create an absorption line Voigt profile model and add it to the
        compound spectral model.

        Parameters
        ----------
        See `~spectacle.modeling.profiles.TauProfile` for method arguments.

        Returns
        -------
        : `~spectacle.modeling.profiles.TauProfile`
            The new tau profile line model.
        """
        kwargs.setdefault('lambda_0', self._center if name is None else None)

        tau_prof = TauProfile(name=name, *args, **kwargs) if model is None else model

        self._line_model = tau_prof if self._line_model is None \
            else self._line_model + tau_prof

        return tau_prof

    @property
    def lsf(self):
        """
        Reads the current LSF model used in the compound spectrum model.

        Returns
        -------
        : `~astropy.modeling.modeling.Fittable1D`
            LSF kernel model.
        """
        return self._lsf_model

    @lsf.setter
    def lsf(self, value):
        """
        Sets the LSF model to be used in the compound spectrum model.

        Parameters
        ----------
        value : `~astropy.modeling.modeling.Fittable1D`
            The model to use. Must take 'y' as input and give 'y' as output.
        """
        if issubclass(value.__class__, Fittable1DModel) or value is None:
            self._lsf_model = value
        else:
            raise ValueError("LSF model must be a subclass of `Fittable1DModel`.")

    def resample(self, x):
        """
        Returns a new `~spectacle.core.spectrum.Spectrum1D` model with a
        resample matrix model as part of the compound spectrum model.

        Returns
        -------
        x : array-like
            Dispersion to which the model will be resampled.
        """
        self._resample_model = Resample()

    @property
    def noise(self):
        """
        Reads the current noise model used in the compound spectrum model.

        Returns
        -------
        : `~astropy.modeling.modeling.Fittable1D`
            Noise model.
        """
        return self._noise_model

    @noise.setter
    def noise(self, value):
        """
        Sets the LSF model to be used in the compound spectrum model.

        Parameters
        ----------
        value : `~astropy.modeling.modeling.Fittable1D`
            The model to use. Must take 'y' as input and give 'y' as output.
        """
        if issubclass(value.__class__, Fittable1DModel) or value is None:
            self._noise_model = value
        else:
            logging.error("Noise model must be a subclass of `Fittable1DModel`.")

    @property
    def optical_depth(self):
        """
        Compound spectrum model in tau space.
        """
        dc = DispersionConvert(self._center)
        rs = self._redshift_model.inverse
        ss = SmartScale(
            1. / (1 + self._redshift_model.z),
            fixed={'factor': True})
        lm = self._line_model

        comp_mod = (rs | lm | ss) if lm is not None else (dc | rs | ss)

        if self.noise is not None:
            comp_mod = comp_mod | self.noise
        if self.lsf is not None:
            comp_mod = comp_mod | self.lsf

        return model_factory(comp_mod, self._center, name="TauModel")()

    @property
    def flux(self):
        """
        Compound spectrum model in flux space.
        """
        dc = DispersionConvert(self._center)
        rs = self._redshift_model.inverse
        ss = SmartScale(
            1. / (1 + self._redshift_model.z),
            fixed={'factor': True})
        cm = self._continuum_model
        lm = self._line_model
        fc = FluxConvert()

        comp_mod = (rs | (cm + (lm | ss | fc))) if lm is not None else (rs | cm + (ss | fc))

        if self.noise is not None:
            comp_mod = comp_mod | self.noise
        if self.lsf is not None:
            comp_mod = comp_mod | self.lsf

        return model_factory(comp_mod, self._center, name="FluxModel")()

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
        ss = SmartScale(
            1. / (1 + self._redshift_model.z),
            fixed={'factor': True})

        comp_mod = (rs | (cm + (lm | ss | fd))) if lm is not None else (dc | rs | cm | ss)

        if self.noise is not None:
            comp_mod = comp_mod | self.noise
        if self.lsf is not None:
            comp_mod = comp_mod | self.lsf

        return model_factory(comp_mod, self._center, name="FluxDecrementModel")()


def model_factory(bases, center, name="BaseModel"):
    from collections import OrderedDict


    class BaseSpectrumModel(bases.__class__):
        inputs = ('x',)
        outputs = ('y',)

        input_units_strict = True
        input_units_allow_dimensionless = {'x': True}

        input_units = {'x': u.AA}
        input_units_equivalencies = {'x': wave_to_vel_equiv(center)}

        @property
        def _supports_unit_fitting(self):
            return True

    return BaseSpectrumModel.rename(name)
