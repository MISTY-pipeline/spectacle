import os
import numpy as np

from astropy.table import Table
from astropy.convolution import Gaussian1DKernel, Kernel1D


class LSF(object):
    """
    Line spread function.
    """
    def __init__(self, function=None, instrument=None, filename=None, *args,
                 **kwargs):
        if instrument is not None:
            if instrument == 'cos':
                self.kernel = COSKernel1D()
        elif filename is not None:
            pass
        elif function is not None:
            if function == 'gaussian':
                self.kernel = Gaussian1DKernel(*args, **kwargs)
            else:
                raise NotImplementedError("LSF using {} is not "
                                          "implemented.".format(function))


class COSKernel1D(Kernel1D):
    """
    1D kernel filter for the COS instrument.
    """
    _separable = True
    _is_bool = False

    def __init__(self, *args, **kwargs):
        path = os.path.abspath(
            os.path.join(__file__, '..', '..', 'data', 'lsfs', 'cos.ecsv'))
        table = Table.read(path, format='ascii')

        super(COSKernel1D, self).__init__(array=table['value'], *args,
                                          **kwargs)