import numpy as np


def _setattr(instance, pname, value):
    """
    Sets parameter value by mapping parameter name to model type.
    Prevents the parameter value setting to be stopped on its tracks
    by non-existent model names or parameter names.
    Parameters
    ----------
    instance: `~astropy.modeling.models`
        The model to initialize.
    mname: str
        Model name.
    pname: str
        Parameter name.
    value: any
        The value to assign.
    """
    # this has to handle both Quantities and plain floats
    try:
        setattr(instance, pname, value.value)
    except AttributeError:
        setattr(instance, pname, value)
    except KeyError:
        pass


class Voigt1DInitializer(object):
    """
    Initialization that is applicable to all "line profile"
    models.
    A "line profile" model is one that has an amplitude, a width,
    and a defined position in wavelength space.
    Parameters
    ----------
    factor: float
        The scale factor to apply to the amplitutde
    """
    def __init__(self, factor=1.0):
        self._factor = factor

    def initialize(self, instance, x, y):
        """
        Initialize the model
        Parameters
        ----------
        instance: `~astropy.modeling.models`
            The model to initialize.
        x, y: numpy.ndarray
            The data to use to initialize from.
        Returns
        -------
        instance: `~astropy.modeling.models`
            The initialized model.
        """

        # X centroid estimates the position
        centroid = np.sum(x * y) / np.sum(y)

        # width can be estimated by the weighted
        # 2nd moment of the X coordinate.
        dx = x - np.mean(x)
        fwhm = 2 * np.sqrt(np.sum((dx * dx) * y) / np.sum(y))

        # amplitude is derived from area.
        delta_x = x[1:] - x[:-1]
        sum_y = np.sum((y[1:]) * delta_x)
        height = sum_y / (fwhm / 2.355 * np.sqrt( 2 * np.pi))

        # Estimate the doppler b parameter
        v_dop = 0.60056120439322491 * fwhm

        # Estimate the column density
        col_dens = (np.trapz(line(self._x.value), self._x.value) * u.Unit(
                'kg/(s2 * Angstrom)') * SIGMA * f_value * center.to('Angstrom')).to('1/cm2')

        _setattr(instance, 'amplitude_L', height * self._factor)
        _setattr(instance, 'x_0', centroid)
        _setattr(instance, 'fwhm_G', fwhm)

        return instance
