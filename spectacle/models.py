import numpy as np

from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.models import RedshiftScaleFactor, Scale, Linear1D
from astropy.modeling import models
import astropy.units as u
from astropy.convolution import Gaussian1DKernel, Kernel1D, convolve

from .modeling import OpticalDepth1DModel, lsfs
from .modeling.lsfs import LSFModel, Gaussian1DKernel
from .modeling.noise import NoiseSTDModel
from .io.registries import line_registry


class Spectrum1DModelMeta(type):
    def __call__(cls, *args, **kwargs):
        if kwargs.get('lines') is not None:
            compound_line_model = np.sum(kwargs.get('lines'))


class NullProfile(Fittable1DModel):
    pass


_Spectrum1DModelBase = type(
    (RedshiftScaleFactor(fixed={'z': True}) |
     NullProfile |
     Scale(tied={'factor': lambda x: 1 / (x.z)}) |
     LSFModel(fixed={'stddev': True}) |
     NoiseSTDModel(fixed={'stddev': True}),
    ), {})


class Spectrum1DModel(_Spectrum1DModelBase):
    inputs = ('x',)
    outputs = ('y',)

    input_units = {'x': u.Unit('km/s')}
    input_units_strict = True
    input_units_allow_dimensionless = True

    @u.quantity_input(rest_wavelength=u.Unit('Angstrom'))
    def __init__(self, ion_name=None, rest_wavelength=None, redshift=None,
                 *args, **kwargs):
        super(Spectrum1DModel, self).__init__(*args, **kwargs)

        # Update the submodel's redshift information
        self.redshift = redshift

        self._rest_wavelength = u.Quantity(rest_wavelength or 0, 'Angstrom')

        if ion_name is not None:
            ion = line_registry.with_name(ion_name)
            self._rest_wavelength = ion['wave'] * line_registry['wave'].unit

    @property
    def redshift(self):
        return self.redshift_submodel.z

    @redshift.setter
    def redshift(self, value):
        self.redshift_submodel.z = value

    @property
    def redshift_submodel(self):
        return next((x for x in self if isinstance(x, RedshiftScaleFactor)))

    @property
    def scale_submodel(self):
        return next((x for x in self if isinstance(x, Scale)))

    @property
    def lsf_submodel(self):
        return next((x for x in self if isinstance(x, LSFModel)))

    @property
    def noise_submodel(self):
        return next((x for x in self if isinstance(x, NoiseSTDModel)))

    @property
    def lines_submodel(self):
        return np.sum([x for x in self if isinstance(x, OpticalDepth1DModel)])

    def add_line(self, **kwargs):
        line = OpticalDepth1DModel(**kwargs)

        mount_point = next((x for x in self if isinstance(x, NullProfile)))

