import astropy.units as u
import numpy as np
from astropy.modeling import Fittable1DModel, Parameter


class FluxConvert(Fittable1DModel):
    inputs = ('y',)
    outputs = ('y',)

    @staticmethod
    def evaluate(y):
        return np.exp(-y) - 1


class FluxDecrementConvert(Fittable1DModel):
    inputs = ('y',)
    outputs = ('y',)

    @staticmethod
    def evaluate(y):
        return 1 - np.exp(-y) - 1