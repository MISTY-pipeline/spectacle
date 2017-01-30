from ..core.registries import line_registry
from ..modeling.models import Voigt1D
from ..core.utils import find_nearest

import numpy as np
import logging


class Line(Voigt1D):
    """
    Data class encapsulating the absorption line feature.
    """
    def __init__(self, name, v_doppler, column_density, lambda_0=None,
                 f_value=None, gamma=None, delta_v=None, delta_lambda=None):
        ind = find_nearest(line_registry['wave'], lambda_0)

        f_value = line_registry['osc_str'][ind]
        lambda_0 = line_registry['wave'][ind]
        gamma_val = line_registry['gamma'][ind]

        super(Line, self).__init__(lambda_0=lambda_0,
                                   f_value=f_value,
                                   gamma=gamma or 0,
                                   v_doppler=v_doppler,
                                   column_density=column_density,
                                   delta_v=delta_v,
                                   delta_lambda=delta_lambda,
                                   name=name)

        # If gamma has not been explicitly defined, tie it to lambda
        if gamma is None:
            self.gamma.value = gamma_val
            self.gamma.tied = lambda cmod, mod=self: _tie_gamma(cmod, mod)


def _tie_gamma(compound_model, model):
    # Find the index of the original model in the compound model
    mod_ind = compound_model._submodels.index(model)

    # The auto-generated name of the parameter in the compound model
    param_name = "lambda_0_{}".format(mod_ind)
    lambda_val = getattr(compound_model, param_name).value

    ind = find_nearest(line_registry['wave'], lambda_val)
    gamma_val = line_registry['gamma'][ind]

    return gamma_val