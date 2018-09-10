import logging
from collections import OrderedDict
from copy import deepcopy

import astropy.units as u
import numpy as np
from astropy.constants import c
from astropy.modeling import Fittable1DModel
from astropy.modeling.models import Const1D
from astropy.table import QTable, Row

from ..analysis import statistics as stats
from ..analysis.resample import Resample
from ..io.registries import line_registry
from ..modeling.converters import (DispersionConvert, FluxConvert,
                                   FluxDecrementConvert)
from ..modeling.custom import RedshiftScaleFactor, Scale
from ..modeling.profiles import OpticalDepth1DModel
from ..utils import wave_to_vel_equiv

dop_rel_equiv = u.equivalencies.doppler_relativistic


class SpectrumModelNotImplemented(Exception):
    pass


class InappropriateModel(Exception):
    pass


class NoLines(Exception):
    pass


class Spectrum1DModel:
    @u.quantity_input(rest_wavelength=u.Unit('Angstrom'))
    def __init__(self, rest_wavelength=None, ion_name=None, redshift=None,
                 continuum=None):
        self._rest_wavelength = u.Quantity(rest_wavelength or 0, 'Angstrom')

        if ion_name is not None:
            ion_name = line_registry.with_name(ion_name)
            self._rest_wavelength = ion_name['wave'] * line_registry['wave'].unit

        self._redshift_model = RedshiftScaleFactor(z=redshift, fixed={'z': True})
        # self._redshift_model._parameter_units_for_data_units = lambda input_units, output_units: dict()

        if continuum is not None and isinstance(continuum, Fittable1DModel):
            self._continuum_model = continuum
        else:
            self._continuum_model = Const1D(1, fixed={'amplitude': True})

            logging.debug("Default continuum set to a 'Constant' model.")

        self._regions = {}
        self._bounds = []
        self._line_model = None
        self._lsf_model = None
        self._noise_model = None
        self._resample_model = None

    def copy(self):
        return deepcopy(self)

    @property
    def rest_wavelength(self):
        """
        The central wavelength value.
        """
        return self._rest_wavelength

    @rest_wavelength.setter
    def rest_wavelength(self, value):
        """
        Define the center wavelength for this spectrum model. The center
        dictates wavelength to velocity space dispersion conversions.

        Parameters
        ----------
        value : `~astropy.units.Quantity`
            Quantity object containing the center value and a unit of type
            *length*.
        """
        self._rest_wavelength = value

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
        self._redshift_model = RedshiftScaleFactor(z=value)

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
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        self._bounds = value

    @property
    def lines(self):
        tab = QTable(names=['name'] + list(OpticalDepth1DModel.param_names),
                     dtype=['S10'] + ['f8'] * len(OpticalDepth1DModel.param_names))

        for l in self.line_models:
            tab.add_row([l.name] + list(l.parameters))

        params = [getattr(OpticalDepth1DModel, n) for n in OpticalDepth1DModel.param_names]

        for i, n in enumerate(OpticalDepth1DModel.param_names):
            tab[n].unit = params[i].unit

        return tab

    @u.quantity_input(x=['length', 'speed'])
    def stats(self, x):
        tab = QTable(names=['name', 'wave', 'col_dens', 'v_dop',
                            'delta_v', 'ew', 'dv90', 'fwhm'],
                     dtype=('S10', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))

        tab['wave'].unit = u.AA
        tab['v_dop'].unit = u.km / u.s
        tab['ew'].unit = u.AA
        tab['dv90'].unit = u.km / u.s
        tab['fwhm'].unit = u.AA
        tab['delta_v'].unit = u.km / u.s

        with u.set_enabled_equivalencies(dop_rel_equiv(self.rest_wavelength)):
            vel = self._redshift_model(self._redshift_model.inverse(x).to('km/s'))
            wav = self._redshift_model(self._redshift_model.inverse(x).to('Angstrom'))

        for sl in self.single_line_spectra:
            line = sl.line_model

            ew = stats.equivalent_width(wav, sl.flux(wav), continuum=self._continuum_model(wav))
            dv90 = stats.delta_v_90(vel, sl.flux_decrement(vel))
            fwhm = line.fwhm(self._redshift_model.inverse(wav))

            tab.add_row([line.name,
                         line.lambda_0,
                         line.column_density,
                         line.v_doppler,
                         line.delta_v,
                         ew,
                         dv90,
                         fwhm])

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
    def single_line_spectra(self):
        single_lines = []

        for l in self.line_models:
            spec = self.copy()
            spec._line_model = l
            single_lines.append(spec)

        return single_lines

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
        kwargs.setdefault('lambda_0', self._rest_wavelength if name is None else None)

        tau_prof = OpticalDepth1DModel(name=name, *args, **kwargs) if model is None else model

        self._line_model = tau_prof if self._line_model is None \
            else self._line_model + tau_prof

        return tau_prof

    def auto_remove_lines(self, condition):
        self._line_model = np.sum([x for x in self.line_model if condition(x)])

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
        rs = self._redshift_model.inverse
        dc = DispersionConvert(self.rest_wavelength)
        ss = Scale(1. / (1 + self.redshift), fixed={'factor': True})
        lm = self._line_model

        comp_mod = rs | dc | (lm | ss) if lm is not None else rs | dc

        if self.noise is not None:
            comp_mod = comp_mod | self.noise
        if self.lsf is not None:
            comp_mod = comp_mod | self.lsf

        return type('OpticalDepth1DModel',
                    (comp_mod.__class__,),
                    {})()

    @property
    def flux(self):
        """
        Compound spectrum model in flux space.
        """
        rs = self._redshift_model.inverse
        dc = DispersionConvert(self.rest_wavelength)
        ss = Scale(1. / (1 + self.redshift), fixed={'factor': True})
        cm = self._continuum_model
        lm = self._line_model
        fc = FluxConvert()

        comp_mod = rs | dc | (cm + (lm | ss | fc)) if lm is not None else rs | dc | cm | fc

        if self.noise is not None:
            comp_mod = comp_mod | self.noise
        if self.lsf is not None:
            comp_mod = comp_mod | self.lsf

        return type('Flux1DModel',
                    (comp_mod.__class__,),
                    {})()

    @property
    def flux_decrement(self):
        """
        Compound spectrum model in flux decrement space.
        """
        rs = self._redshift_model.inverse
        dc = DispersionConvert(self.rest_wavelength)
        ss = Scale(1. / (1 + self.redshift), fixed={'factor': True})
        cm = self._continuum_model
        lm = self._line_model
        fd = FluxDecrementConvert()

        comp_mod = rs | dc | (cm + (lm | ss | fd)) if lm is not None else rs | dc | cm | fd

        if self.noise is not None:
            comp_mod = comp_mod | self.noise
        if self.lsf is not None:
            comp_mod = comp_mod | self.lsf

        return type('FluxDecrement1DModel',
                    (comp_mod.__class__,),
                    {})()
