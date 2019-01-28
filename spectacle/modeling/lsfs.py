import os

from astropy.convolution import Gaussian1DKernel, Kernel1D, convolve
from astropy.modeling import Fittable1DModel
from astropy.table import Table

__all__ = ['COSLSFModel', 'GaussianLSFModel']


class COSKernel1D(Kernel1D):
    """
    1D kernel filter for the COS instrument.
    """
    _separable = True
    _is_bool = False

    def __init__(self):
        path = os.path.abspath(
            os.path.join(__file__, '..', '..', 'data', 'cos_lsf.ecsv'))
        table = Table.read(path, format='ascii.ecsv')

        super(COSKernel1D, self).__init__(array=table['value'])


class LSFModel(Fittable1DModel):
    inputs = ('y',)
    outputs = ('y',)

    def __init__(self, kernel=None, *args, **kwargs):
        super(LSFModel, self).__init__(*args, **kwargs)

        self._kernel = kernel

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, value):
        self._kernel = value

    def evaluate(self, y, *args, **kwargs):
        # TODO: why is the y array including an extra dimesion?
        if y.ndim > 1:
            y = y[0]

        return convolve(y, self.kernel, boundary='extend')


class COSLSFModel(LSFModel):
    """
    COS LSF model which can be used with the compound model objects.
    """
    inputs = ('y',)
    outputs = ('y',)

    def __init__(self, *args, **kwargs):
        super().__init__(kernel=COSKernel1D(), *args, **kwargs)


class GaussianLSFModel(LSFModel):
    """
    Gaussian LSF model which can used with the compound model objects.
    """
    inputs = ('y',)
    outputs = ('y',)

    def __init__(self, stddev, *args, **kwargs):
        super().__init__(kernel=Gaussian1DKernel(stddev), *args, **kwargs)

        self._kernel_args = args
        self._kernel_kwargs = kwargs
