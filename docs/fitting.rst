.. _fitting:

Fitting
=======

The core :class:`~spectacle.modeling.models.Spectral1D` behaves exactly like
an Astropy model and can be used with any of the supported non-linear
Astropy fitters, as well as some not included in the Astropy library.

The most common one might use is the
:class:`~astropy.modeling.fitting.LevMarLSQFitter`:

.. plot::
    :include-source:
    :align: center
    :context: close-figs

    >>> from astropy.modeling.fitting import LevMarLSQFitter
    >>> from spectacle.modeling import Spectral1D, OpticalDepth1D
    >>> import astropy.units as u
    >>> from matplotlib import pyplot as plt
    >>> import numpy as np

    Generate some fake data to fit to:

    >>> line1 = OpticalDepth1D("HI1216", v_doppler=10 * u.km/u.s, column_density=14)
    >>> spec_mod = Spectral1D(line1, continuum=1)
    >>> x = np.linspace(-200, 200, 1000) * u.Unit('km/s')
    >>> y = spec_mod(x) + (np.random.sample(1000) - 0.5) * 0.01

    Instantiate the fitter and fit the model to the data:

    >>> fitter = LevMarLSQFitter()
    >>> fit_spec_mod = fitter(spec_mod, x, y)

    Users can see the results of the fitted spectrum by printing the returned
    model object

    >>> print(fit_spec_mod)
    Model: Spectral1D
    Inputs: ('x',)
    Outputs: ('y',)
    Model set size: 1
    Parameters:
        amplitude_0 z_1 lambda_0_2 f_value_2   gamma_2      v_doppler_2      column_density_2      delta_v_2          delta_lambda_2    z_4
                         Angstrom                              km / s                                km / s              Angstrom
        ----------- --- ---------- --------- ----------- ------------------ ------------------ ------------------ --------------------- ---
                1.0 0.0  1215.6701    0.4164 626500000.0 10.010182187404824 13.998761432240995 1.0052009119192702 -0.004063271434522016 0.0

    Plot the results:

    >>> plt.step(x, y, label="Data")
    >>> plt.step(x, fit_spec_mod(x), label="Fit")
    >>> plt.legend()


Fitting with the line finder
----------------------------

The :class:`~spectacle.fitting.line_finder.LineFinder1D` class can also be
passed a fitter instance if the user wishes to use a specific type as opposed to the
default Levenberg-Marquardt algorithm.


.. code-block:: python
    :linenos:

    line_finder = LineFinder1D(ions=["HI1216", "OVI1038"], continuum=0, output='optical_depth', fitter=LevMarLSQFitter())
