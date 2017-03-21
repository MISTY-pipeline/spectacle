from ..core.registries import line_registry
from ..modeling.models import Voigt1D
from ..core.utils import find_nearest

import numpy as np
import logging
from uncertainties import unumpy as unp


class Line(Voigt1D):
    """
    Data class encapsulating the absorption line feature.
    """
    def __init__(self, name, v_doppler, column_density, lambda_0=None,
                 f_value=None, gamma=None, delta_v=None, delta_lambda=None,
                 tied=None):
        if f_value is None:
            if lambda_0 is None:
                lind = np.min(np.where(line_registry['name'] == name))
                lambda_0 = line_registry['wave'][lind]

            ind = find_nearest(line_registry['wave'], lambda_0)
            f_value = line_registry['osc_str'][ind]

        super(Line, self).__init__(lambda_0=lambda_0,
                                   f_value=f_value,
                                   gamma=gamma or 0,
                                   v_doppler=v_doppler,
                                   column_density=column_density,
                                   delta_v=delta_v,
                                   delta_lambda=delta_lambda,
                                   name=name)

        # If gamma has not been explicitly defined, tie it to lambda
        if tied is not None:
            if 'gamma' in tied:
                self.gamma.tied = lambda cmod, mod=self: _tie_gamma(cmod, mod)

    @property
    def fwhm(self):
        """
        Calculates an approximation of the FWHM.

        The approximation is accurate to
        about 0.03% (see http://en.wikipedia.org/wiki/Voigt_profile).

        Returns
        -------
        fwhm : float
            The estimate of the FWHM
        """
        # The width of the Lorentz profile
        fl = 2.0 * self.gamma

        # Width of the Gaussian [2.35 = 2*sigma*sqrt(2*ln(2))]
        fd = 2.35482 * 1/np.sqrt(2.)

        fwhm = 0.5346 * fl + np.sqrt(0.2166 * (fl ** 2.) + fd ** 2.)

        return fwhm


def _tie_gamma(compound_model, model):
    # Find the index of the original model in the compound model
    mod_ind = compound_model._submodels.index(model)

    # The auto-generated name of the parameter in the compound model
    param_name = "lambda_0_{}".format(mod_ind)
    lambda_val = getattr(compound_model, param_name).value

    ind = find_nearest(line_registry['wave'], lambda_val)
    gamma_val = line_registry['gamma'][ind]

    return gamma_val