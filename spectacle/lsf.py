from astropy.convolution import Gaussian1DKernel


class LSF(object):
    """
    Line spread function.
    """
    def __init__(self, function, *args, **kwargs):
        if function == 'gaussian':
            self.kernel = Gaussian1DKernel(*args, **kwargs)
        else:
            raise NotImplementedError("LSF using {} is not "
                                      "implemented.".format(function))