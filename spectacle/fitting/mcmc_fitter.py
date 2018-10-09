from abc import ABCMeta, abstractmethod
from astropy.modeling.fitting import _fitter_to_model_params
from astropy.modeling.models import Gaussian1D
import numpy as np


class ObjectiveFunction(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class LogLikelihood(ObjectiveFunction):
    def __init__(self, model, x, y):
        self._model = model
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def model(self):
        return self._model

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)


class GaussianLogLikelihood(LogLikelihood):
    def __init__(self, *args, yerr, **kwargs):
        super().__init__(*args, **kwargs)

        self._yerr = yerr

    @property
    def yerr(self):
        return self._yerr

    def evaluate(self, *args, **kwargs):
        _fitter_to_model_params(self.model, list(args) + list(kwargs.values()))
        mean_model = self.model(self.x)

        lnlike = np.sum(-0.5 * np.log(2 * np.pi) - np.log(self.yerr) -
                        (self.y - mean_model) ** 2 / (2 * self.yerr ** 2))

        return lnlike

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)


class LogPosterior(ObjectiveFunction):
    def __init__(self, model, x, y):
        self._model = model
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def model(self):
        return self._model

    @abstractmethod
    def lnlikelihood(self, *args, **kwargs):
        pass

    @abstractmethod
    def lnprior(self, *args, **kwargs):
        pass

    def lnposterior(self, *args, **kwargs):
        return self.lnprior(*args, **kwargs) + self.lnlikelihood(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.lnposterior(*args, **kwargs)


class GaussianLogPosterior(LogPosterior):
    def __init__(self, *args, yerr, **kwargs):
        super().__init__(*args, **kwargs)

        self._yerr = yerr

    @property
    def yerr(self):
        return self._yerr

    def lnprior(self, *args, **kwargs):
        parameters = list(args) + list(kwargs.values())

    def lnlikelihood(self, *args, **kwargs):
        _fitter_to_model_params(self.model, list(args) + list(kwargs.values()))

        mean_model = self.model(self.x)

        lnlike = np.sum(-0.5 * np.log(2 * np.pi) - np.log(self.yerr) -
                        (self.y - mean_model) ** 2 / (2 * self.yerr ** 2))

        return lnlike

