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
    def __init__(self, name, v_doppler=None, column_density=None,
                 lambda_0=None, f_value=None, gamma=None, delta_v=None,
                 delta_lambda=None, tied=None, fixed=None):
        if lambda_0 is None:
            if name not in line_registry['name']:
                logging.error("No ion named {} in line registry.".format(name))
                return

            lind = np.min(np.where(line_registry['name'] == name))
            lambda_0 = line_registry['wave'][lind]

        if f_value is None:
            ind = find_nearest(line_registry['wave'], lambda_0)
            f_value = line_registry['osc_str'][ind]

        if gamma is None:
            ind = find_nearest(line_registry['wave'], lambda_0)
            gamma = line_registry['gamma'][ind]
            tied = {'gamma': lambda cmod, mod=self: _tie_gamma(cmod, mod)}

            logging.info("Gamma is being tied to values within your ion"
                         "lookup table.")

        super(Line, self).__init__(lambda_0=lambda_0,
                                   f_value=f_value,
                                   gamma=gamma or 0,
                                   v_doppler=v_doppler,
                                   column_density=column_density,
                                   delta_v=delta_v,
                                   delta_lambda=delta_lambda,
                                   name=name,
                                   tied=tied or {},
                                   fixed=fixed or {})

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
    mod = next((x for x in compound_model._submodels[1:]
                if _compare_models(x, model)), None)

    mod_ind = [x for x in compound_model._submodels].index(mod)

    # The auto-generated name of the parameter in the compound model
    param_name = "lambda_0_{}".format(mod_ind)
    lambda_val = getattr(compound_model, param_name).value

    ind = find_nearest(line_registry['wave'], lambda_val)
    gamma_val = line_registry['gamma'][ind]

    return gamma_val

def _compare_models(mod1, mod2):
    """
    Check to see if two models are functionally equivalent.
    """
    attrs = ['lambda_0', 'f_value', 'gamma', 'column_density', 'v_doppler',
             'delta_v', 'delta_lambda']

    return all([getattr(mod1, attr).value == getattr(mod2, attr).value
                for attr in attrs])
