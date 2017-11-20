"""
"""
import astropy.units as u
from numpy import ndarray
import numpy as np
from numpy.lib.recfunctions import append_fields

from .models import OpticalDepth, TauProfile


class NDSpectrum:
    """

    """
    @u.quantity_input(
        spectral_axis=['length', 'speed'],
        flux=['spectral flux density wav', 'spectral flux density'])
    def __init__(self, spectral_axis=None, flux=None):
        self._spectral_axis = spectral_axis
        self._flux = flux
        self._absorber_model = TauProfile(
            lambda_0=1215.6701 * u.Angstrom, f_value=0.4164, gamma=626500000.0,
            v_doppler=1e6 * u.Unit('cm/s'),
            column_density=1e15 * u.Unit('1/cm2'), delta_v=0 * u.Unit('km/s'),
            delta_lambda=0 * u.Angstrom)

    @property
    def spectral_axis(self):
        return self._spectral_axis

    @property
    def flux(self):
        return self._flux

    @property
    def absorber_model(self):
        return self._absorber_model

    @u.quantity_input(rest_wave='length')
    def optical_depth(self, rest_wave):
        disp = self.spectral_axis.to('Angstrom',
                                     equivalencies=u.doppler_radio(rest_wave))
        opt_dep_mod = OpticalDepth(self)

        return opt_dep_mod(disp)


class SpectrumND:
    def __init__(self, obj, units=None, meta=None, spectral_axis=None, 
                 *args, **kwargs):
        self._data = np.rec.array(obj, *args, **kwargs)
        self._units = dict.fromkeys(self._data.dtype.names, u.Unit(''))

        # Establish the meta information dictionaries for each name defined in
        # the record array. Update the dictionary with user-provided meta
        # dictionary. Note that this does not check that the user's naming
        # matches the names provided to the record array.
        self._meta = dict.fromkeys(self._data.dtype.names, {})
        self._meta.update(meta or {})

        if isinstance(obj, (list, tuple)):
            self._units.update(
                zip(self._data.dtype.names,
                    map(lambda x: x.unit if isinstance(x, u.Quantity)
                        else u.Unit(''), obj)))

        if isinstance(units, (list, tuple)):
            self._units.update(zip(self._data.dtype.names,
                                   map(lambda x: u.Unit(x), units)))
        elif isinstance(units, dict):
            self._units.update(units)

        if isinstance(spectral_axis, (list, np.ndarray)):
            self._data['spectral_axis'] = spectral_axis
        elif isinstance(spectral_axis, (u.Quantity)):
            self._data['spectral_axis'] = spectral_axis.value
            self._units['spectral_axis'] = spectral_axis.unit

    def __getitem__(self, key):
        arr = self._data.__getitem__(key)

        if isinstance(key, str):
            if key in self._units:
                return u.Quantity(arr, self.units[key])
        # elif isinstance(key, (int, slice)):
        #     return list(map(lambda x: x[0] * x[1],
        #                     zip(arr, self._units.values())))

        return self.new(arr)

    def new(self, arr, *args, **kwargs):
        return self.__class__(arr,
                              units=kwargs.get('units', self.units),
                              meta=kwargs.get('meta', self.meta),
                              names=kwargs.get('names', self.names))

    # def __getattribute__(self, name):
    #     return self._data.__getattribute__(name)

    @property
    def data(self):
        return self._data

    @property
    def units(self):
        return self._units

    @property
    def meta(self):
        return self._meta

    @property
    def names(self):
        return self._data.dtype.names

    def append(self, field_name, field_value, unit=None, *args, **kwargs):
        new_rec = append_fields(self._data, field_name, field_value,
                                asrecarray=True, usemask=False)

        if isinstance(field_value, u.Quantity) or \
                issubclass(field_value.__class__, u.Quantity):
            unit = field_value.unit or unit

        units = self.units.copy()
        units[field_name] = unit

        return self.new(new_rec, units=units)

    def convert(self, field_name, new_unit, equivalencies=None):
        quant = self[field_name]
        self._data[field_name] = quant.to(
            new_unit, equivalencies=equivalencies).value
        self._units[field_name] = new_unit
