===============
Getting Started
===============

Spectacle is designed to facilitate the creation of complex spectral models.
It is built on the `Astropy modeling <http://docs.astropy.org/en/stable/modeling/>`_ module.
It can create representations of spectral data in wavelength, frequency, or
velocity space, with any number of line components. These models can then be
fit to data for use in individual ion reduction, or characterization of spectra
as a whole.

Creating a spectral model
-------------------------

The primary class is the :class:`~spectacle.modeling.models.Spectral1D`. This
serves as the central object through which the user will design their model,
and so exposes many parameters to the user.

:class:`~spectacle.modeling.models.Spectral1D` models are initialized with
several descriptive properties:

+--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Attribute          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
+====================+============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================+
| `z`                | This is the redshift of the spectrum. It can either be a single numerical value, or an array describing the redshift at each e.g. velocity bin. Note if the user provides an array, the length of the redshift array must match the length of the dispersion array used as input to the model. *Default: 0*                                                                                                                                                                                                                                |
+--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `rest_wavelength`  | The wavelength at which transformations from wavelength space to velocity space will be performed. *Default: 0 Angstrom*                                                                                                                                                                                                                                                                                                                                                                                                                   |
+--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `output`           | Describes the type of data the spectrum model will produce. This can be one of `flux`, `flux_decrement`, or `optical_depth`. *Default: `flux`*                                                                                                                                                                                                                                                                                                                                                                                             |
+--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `continuum`        | A :class:`~astropy.modeling.models.FittableModel1D` or single numeric value  representing the continuum for the spectral model. *Default: 0*                                                                                                                                                                                                                                                                                                                                                                                               |
+--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `lsf`              | The line spread function to be applied to the model. Users can provided an :class:`~spectacle.modeling.lsfs.LSFModel`, a :class:`~astropy.convolution.Kernel1D`, or a string indicating either `cos` for the COS LSF, or `guassian` for a Gaussian LSF. In the latter case, the user should provide key word arguments as parameters for the Gaussian profile.                                                                                                                                                                             |
+--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `lines`            | The set of lines to be added to the spectrum. This information is passed to the :class:`~spectacle.modeling.profiles.OpticalDepth1D`  initializer. This can either be a single :class:`~spectacle.modeling.profiles.OpticalDepth1D` instance, a list of :class:`~spectacle.modeling.profiles.OpticalDepth1D` instances; a single string representing the name of an ion (e.g. "HI1216"), a list of such strings; a single :class:`~astropy.units.Quantity` value representing the rest wavelength of an ion, or a list of such values.     |
+--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

The :class:`~spectacle.modeling.models.Spectral1D` does not require that any
lines be provided initially. In that case, it will just generate data only
considering the continuum and other properties.

Providing lines to the initializer
----------------------------------

As mentioned in the table above, lines can be added by passing a set of
:class:`~spectacle.modeling.models.OpticalDepth1D` instances, ion name strings,
or rest wavelength :class:`~astropy.units.Quantity` objects to the initializer.

Using a set of :class:`~spectacle.modeling.models.OpticalDepth1D` instances::

    line = OpticalDepth1D("HI1216", v_doppler=50 * u.km/u.s, column_density=14)
    line2 = OpticalDepth1D("HI1216", delta_v=100 * u.km/u.s)
    spec_mod = Spectral1D([line, line2])

Using ion name strings::

    spec_mod = Spectral1D(["HI1216", "OVI1032"])

Using rest wavelength :class:`~astropy.units.Quantity` objects::

    spec_mod = Spectral1D([1216 * u.Angstrom, 1032 * u.Angstrom])

Adding lines after model creation
---------------------------------

Likewise, the user can add a line to an already made spectral model by using
the :meth:`~spectacle.modeling.models.Spectral1D.with_line` method, and
provide to it information accepted by the :class:`~spectacle.modeling.models.OpticalDepth1D`
class

.. code-block:: python

    >>> from spectacle import Spectral1D
    >>> import astropy.units as u
    >>> spec_mod = Spectral1D([1216 * u.AA])
    >>> spec_mod = spec_mod.with_line("HI1216", v_doppler=50 * u.km/u.s, column_density=14)
    >>> print(spec_mod)  # doctest: +SKIP
    Model: Spectral1D
    Inputs: ('x',)
    Outputs: ('y',)
    Model set size: 1
    Parameters:
        amplitude_0 z_1 lambda_0_2 f_value_2   gamma_2   v_doppler_2 column_density_2 delta_v_2 delta_lambda_2 lambda_0_3 f_value_3   gamma_3   v_doppler_3 column_density_3 delta_v_3 delta_lambda_3 z_5
                         Angstrom                           km / s                      km / s     Angstrom     Angstrom                           km / s                      km / s     Angstrom
        ----------- --- ---------- --------- ----------- ----------- ---------------- --------- -------------- ---------- --------- ----------- ----------- ---------------- --------- -------------- ---
                0.0 0.0  1215.6701    0.4164 626500000.0        10.0             13.0       0.0            0.0  1215.6701    0.4164 626500000.0        50.0             14.0       0.0            0.0 0.0


