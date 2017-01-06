import six
from astropy.modeling import Fittable1DModel
from astropy.modeling.models import Linear1D
from ..core.models import Voigt1D


class SpectrumModelMeta(type):
    def __new__(mcs, name, bases, attr):
        return super(SpectrumModelMeta, mcs).__new__(name, bases, attr)


@six.add_metaclass(SpectrumModelMeta)
class Spectrum1DModel:

    @classmethod
    def add_line(self):
        return