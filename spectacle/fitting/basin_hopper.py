from astropy.modeling.fitting import Fitter, _model_to_fit_params, ModelLinearityError, _fitter_to_model_params, _FitterMeta
import numpy as np

from scipy import optimize


DEFAULT_MAXITER = 100
DEFAULT_EPS = np.sqrt(np.finfo(float).eps)
DEFAULT_MAX_BOUND = 10 ** 12
DEFAULT_MIN_BOUND = -10 ** 12


class StepTaker:
    def __init__(self, stepsize=0.001, bounds=None):
        self.stepsize = stepsize

    def __call__(self, x):
        x[0] += np.random.lognormal(self.stepsize, self.stepsize)
        x[1] += np.random.uniform(-10 * self.stepsize, 10 * self.stepsize)

        return x


class BasinHopperFitter(metaclass=_FitterMeta):
    supported_constraints = ['bounds', 'eqcons', 'ineqcons', 'fixed', 'tied']

    def __init__(self, *args, bounds=None, **kwargs):
        super().__init__(*args, **kwargs)

        self._previous_x = None
        self._step_sizes = np.array([.1, 10])

        if bounds is not None:
            for idx, bnds in enumerate(bounds):
                step = (bnds[1] - bnds[0]) * 0.1
                self._step_sizes[idx] = np.random.uniform(-step, step)

        self._minimum = np.inf

    def objective_function(self, fps, *args):
        model, x, y, weights = args
        _fitter_to_model_params(model, fps)
        res = model(x) - y

        return np.sum(res ** 2)

    def _take_step(self, x, bounds):
        x += np.random.uniform(-self._step_sizes, self._step_sizes, x.shape)

        # for idx, bnds in enumerate(bounds):
        #     step = (bnds[1] - bnds[0]) * 0.1
        #     x[idx] = np.random.uniform(-step, step)

        return x

    def _dynamic_step(self, x, f, acc):
        if not acc:
            return
        print(f, x, acc)

        if np.isinf(self._minimum):
            self._previous_x = x
            self._minimum = f
        elif f < self._minimum:
            print("Changing step size from ", self._step_sizes)
            self._step_sizes = np.abs(x - self._previous_x)
            print("To ", self._step_sizes)
            self._minimum = f

    @staticmethod
    def _bounds_check(f_new, x_new, f_old, x_old, bounds):
        _bnds_check = []

        for i in range(len(x_new)):
            if bounds[i][0] is None:
                mn_bool = True
            else:
                mn_bool = bounds[i][0] < x_new[i]

            # print("Min check: {}, {}".format(x_new[i], mn_bool))

            if bounds[i][1] is None:
                mx_bool = True
            else:
                mx_bool = bounds[i][1] > x_new[i]

            # print("Max check: {}, {}".format(x_new[i], mx_bool))

            _bnds_check.append(mn_bool and mx_bool)

        return all(_bnds_check)

    def __call__(self, model, x, y, weights=None, maxiter=DEFAULT_MAXITER,
                 epsilon=DEFAULT_EPS):
        if model.linear:
            raise ModelLinearityError(
                'Model is linear in parameters; '
                'non-linear fitting methods should not be used.')

        model_copy = model.copy()
        init_values, fit_param_indicies = _model_to_fit_params(model_copy)
        bounds = np.array(list(model.bounds.values()))[fit_param_indicies]

        minimizer_kwargs = {"method": "BFGS"}
        opt_res = optimize.basinhopping(
            lambda fps: self.objective_function(fps, model_copy, x, y, weights),
            init_values,
            minimizer_kwargs=minimizer_kwargs,
            accept_test=lambda *args, **kwargs: self._bounds_check(*args, bounds=bounds, **kwargs),
            take_step=lambda *args, **kwargs: self._take_step(*args, bounds=bounds, **kwargs),
            callback=lambda x, f, acc: self._dynamic_step(x, f, acc)
            )

        _fitter_to_model_params(model_copy, opt_res.x)

        return model_copy