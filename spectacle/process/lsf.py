import os
import numpy as np
import abc
import six

from astropy.table import Table
from astropy.convolution import Gaussian1DKernel, Kernel1D


@six.add_metaclass(abc.ABCMeta)
class LSF:
    """
    Line spread function.
    """
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class COSLSF(LSF):
    def __init__(self):
        self.kernel = COSKernel1D()

        super(COSLSF, self).__init__()


class GaussianLSF(LSF):
    def __init__(self, *args, **kwargs):
        self.kernel = Gaussian1DKernel(*args, **kwargs)

        super(GaussianLSF, self).__init__()


class COSKernel1D(Kernel1D):
    """
    1D kernel filter for the COS instrument.
    """
    _separable = True
    _is_bool = False

    def __init__(self):
        path = os.path.abspath(
            os.path.join(__file__, '..', '..', 'data', 'lsfs', 'cos.ecsv'))
        table = Table.read(path, format='ascii')

        super(COSKernel1D, self).__init__(array=table['value'])
