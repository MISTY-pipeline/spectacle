import os

from astropy.table import Table
from astropy.convolution import Gaussian1DKernel, Kernel1D, convolve
from astropy.modeling import Fittable1DModel, Parameter


__all__ = ['COSLSFModel', 'GaussianLSFModels']


class COSKernel1D(Kernel1D):
    """
    1D kernel filter for the COS instrument.
    """
    _separable = True
    _is_bool = False

    def __init__(self):
        path = os.path.abspath(
            os.path.join(__file__, '..', 'data', 'cos.ecsv'))
        table = Table.read(path, format='ascii.ecsv')

        super(COSKernel1D, self).__init__(array=table['value'])


class COSLSFModel(Fittable1DModel):
    """
    COS LSF model which can be used with the compound model objects.
    """
    input = ('y',)
    output = ('y',)

    @staticmethod
    def evaluate(y):
        kernel = COSKernel1D()

        return convolve(y, kernel)


class GaussianLSFModel(Fittable1DModel):
    """
    Gaussian LSF model which can used with the compound model objects.
    """
    input = ('y',)
    output = ('y',)

    stddev = Parameter(default=0, min=0, fixed=True)

    @staticmethod
    def evaluate(y, stddev):
        kernel = Gaussian1DKernel(stddev=stddev)

        return convolve(y, kernel)