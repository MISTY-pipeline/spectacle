from collections import OrderedDict

import astropy.units as u
import numpy as np
from astropy.modeling import Fittable1DModel, Fittable2DModel, Parameter
from astropy.modeling.models import RedshiftScaleFactor, Scale, Linear1D

from ..core.region_finder import find_regions
from ..modeling.converters import VelocityConvert, WavelengthConvert
from ..modeling.profiles import TauProfile
from ..utils import wave_to_vel_equiv

__all__ = ['Redshift', 'SmartScale', 'Masker']


class Redshift(RedshiftScaleFactor):
    z = Parameter(default=0, min=0, fixed=True)

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return OrderedDict()


class SmartScale(Scale):
    input_units_strict = True
    input_units_allow_dimensionless = {'x': True}

    factor = Parameter(default=1, min=0, fixed=True)

    @staticmethod
    def evaluate(x, factor):
        """One dimensional Scale model function"""
        if isinstance(factor, u.Quantity):
            return_unit = factor.unit
            factor = factor.value

            if isinstance(x, u.Quantity):
                return (x.value * factor) * return_unit
        else:
            return factor * x

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return OrderedDict()


class Linear(Linear1D):
    input_units_strict = True
    input_units_allow_dimensionless = {'x': True}

    @property
    def input_units(self):
        if self.slope.unit is not None:
            return {'x': 1/self.slope.unit}

        return None

    # def evaluate(self, x, slope, intercept):
    #     x = u.Quantity(x, self.input_units['x'])
    #     slope = u.Quantity(slope, 1/self.input_units['x'])
    #     intercept = u.Quantity(intercept, u.Unit(""))

    #     return super(Linear, self).evaluate(x, slope, intercept)

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return OrderedDict([('slope', 1/inputs_unit['x']),
                            ('intercept', u.Unit(""))])

class Masker(Fittable2DModel):
    """
    Model for masking uninteresting features in a spectrum and dispersion
    array. This class attempts to identify features in a spectrum and use a
    region finding algorithm to define the bounds of the region.

    The continuum can be provided by the user. By default, the algorithm
    assumes the continuum is zero. In the case of a tau profile or flux
    decrement, this is probably the case. However, in the case of flux,
    the user should supply a more relevant continuum that will be
    subtracted from the flux data.

    The line list attribute allows a user to specify a small selection of ions.
    Any identified regions that do not contain the ion's centroid will be
    ignored.
    """
    inputs = ('x', 'y')
    outputs = ('z')
    input_units_strict = True

    # The center parameter should be tied to the compound model's dispersion
    center = Parameter(default=0, fixed=True, unit=u.Unit('Angstrom'))

    def __init__(self, continuum=None, line_list=None, rel_tol=1e-2,
                 abs_tol=1e-4, *args, **kwargs):
        """
        Masker model for identifying regions in a spectrum.

        Parameters
        ----------
        continuum : array-like
            Continuum array that will be subtracted from the spectral data
            array.
        line_list : list
            A list of ion names which will be used to look up centroid
            information in the line registry.
        rel_tol : float
            The relative tolerance between the continuum and the spectral data.
        abs_tol : float
            The absolute tolerance btween the continuum and the spectral data.

        Returns
        -------
        `~np.ma.MaskedArray`
            A masked array of the dispersion data.
        `~np.ma.MaskedArray`
            A masked array of the spectral data.
        """
        super(Masker, self).__init__(*args, **kwargs)
        self._line_list = line_list

        # In the case where the user has provided a list of ion names, attempt
        # to find the ions in the database
        if self._line_list:
            self._line_list = [TauProfile(name=x) for x in self._line_list]

        self._continuum = continuum
        self._rel_tol = rel_tol
        self._abs_tol = abs_tol

    def evaluate(self, x, y):
        continuum = self._continuum if self._continuum is not None else np.zeros(
            y.shape)

        reg = find_regions(y, continuum=continuum, rel_tol=self._rel_tol,
                           abs_tol=self._abs_tol)

        if self._line_list is not None:
            filt_reg = []

            for rl, rr in reg:
                if any([x[rl] <= prof.lambda_0 <= x[rr] for prof in self._line_list]):
                    filt_reg.append((rl, rr))

            reg = filt_reg

        mask = np.logical_or.reduce(
            [(x > x[rl]) & (x <= x[rr]) for rl, rr in reg])

        # Ensure that the output quantities are the original input quantities
        # x = x.to(self.output_units['x'],
        #          equivalencies=self.input_units_equivalencies['x'])

        # return np.ma.array(x, mask=~mask), np.ma.array(y, mask=~mask)
        return mask
