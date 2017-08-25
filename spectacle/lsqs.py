import os

from astropy.table import Table
from astropy.convolution import Gaussian1DKernel, Kernel1D


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
