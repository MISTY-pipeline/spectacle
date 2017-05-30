from ..core.registries import line_registry
from ..modeling.models import Voigt1D
from ..core.utils import find_nearest

import numpy as np
import logging
from astropy import constants as c


class Line(Voigt1D):
    """
    Data class encapsulating the absorption line feature.
    """
    def __init__(self, name, v_doppler=None, column_density=None,
                 lambda_0=None, f_value=None, gamma=None, delta_v=None,
                 delta_lambda=None, tied=None, fixed=None):
        lambda_val = lambda_0 * (1 + (delta_v or 0) / c.c.cgs.value) + \
                     (delta_lambda or 0)
        tied = tied or {}
        fixed = fixed or {}

        if lambda_0 is None:
            if name not in line_registry['name']:
                logging.error("No ion named {} in line registry.".format(name))
                return

            lind = np.min(np.where(line_registry['name'] == name))
            lambda_0 = line_registry['wave'][lind]

        if tied is None:
            if f_value is None and not fixed.get('f_value', True):
                tied.update({
                    'f_value': lambda cmod, mod=self:
                        _tie_nearest(cmod, mod, line_registry['osc_str'])})

            if gamma is None and not fixed.get('f_value', True):
                tied.update({'gamma': lambda cmod, mod=self:
                        _tie_nearest(cmod, mod, line_registry['gamma'])})

        if f_value is None:
            ind = find_nearest(line_registry['wave'], lambda_val)
            f_value = line_registry['osc_str'][ind]

        if gamma is None:
            ind = find_nearest(line_registry['wave'], lambda_val)
            gamma = line_registry['gamma'][ind]

        super(Line, self).__init__(lambda_0=lambda_0,
                                   f_value=f_value,
                                   gamma=gamma or 0,
                                   v_doppler=v_doppler,
                                   column_density=column_density,
                                   delta_v=delta_v,
                                   delta_lambda=delta_lambda,
                                   name=name,
                                   tied=tied,
                                   fixed=fixed)

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


def _tie_nearest(compound_model, model, column):
    # The auto-generated name of the parameter in the compound model
    # param_name = "lambda_0_{}".format(mod_ind)
    lambda_val = model.lambda_0.value
    delta_v = model.delta_v.value
    delta_lambda = model.delta_lambda.value

    # Incorporate shifts of the lambda value
    lambda_val = lambda_val * (1 + delta_v / c.c.cgs.value) + delta_lambda

    ind = find_nearest(line_registry['wave'], lambda_val)
    val = column[ind]

    return val


def _compare_models(mod1, mod2):
    """
    Check to see if two models are functionally equivalent.
    """
    attrs = ['lambda_0', 'f_value', 'gamma', 'column_density', 'v_doppler',
             'delta_v', 'delta_lambda']

    return all([getattr(mod1, attr).value == getattr(mod2, attr).value
                for attr in attrs])
