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

    >>> line1 = OpticalDepth1D(lambda_0=1216 * u.AA, v_doppler=500 * u.km/u.s, column_density=14)
    >>> spec_mod = Spectral1D(line1, continuum=1)
    >>> x = np.linspace(1200, 1225, 1000) * u.Unit('Angstrom')
    >>> y = spec_mod(x) + (np.random.sample(1000) - 0.5) * 0.01

    Instantiate the fitter and fit the model to the data:

    >>> fitter = LevMarLSQFitter()
    >>> fit_spec_mod = fitter(spec_mod, x, y)

    Plot the results:

    >>> f, ax = plt.subplots()  # doctest: +IGNORE_OUTPUT
    >>> ax.step(x, y, label="Data")  # doctest: +IGNORE_OUTPUT
    >>> ax.step(x, fit_spec_mod(x), label="Fit")  # doctest: +IGNORE_OUTPUT
    >>> ax.legend()  # doctest: +IGNORE_OUTPUT


Fitting with the line finder
----------------------------

The :class:`~spectacle.fitting.line_finder.LineFinder1D` class can also be
passed a fitter instance if the user wishes to use a specific type as opposed to the
default Levenberg-Marquardt algorithm.


.. code-block:: python
    :linenos:

    line_finder = LineFinder1D(ions=["HI1216", "OVI1038"], continuum=0, output='optical_depth', fitter=LevMarLSQFitter())
