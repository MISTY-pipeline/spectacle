import os

import numpy as np
from astropy.modeling import Parameter
from astropy.convolution import Gaussian1DKernel, Kernel1D, convolve
from astropy.modeling import Fittable1DModel, Parameter

__all__ = ['NoiseSTDModel']


class NoiseSTDModel(Fittable1DModel):
    """
    Gaussian LSF model which can used with the compound model objects.
    """
    inputs = ('y',)
    outputs = ('y',)
    
    stddev = Parameter(default=0, min=0, fixed=True)

    @staticmethod
    def evaluate(y, stddev):
        noise = np.random.normal(0., stddev, y.size)

        return y + noise
