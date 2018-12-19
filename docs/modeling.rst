Modeling
========

The purpose of Spectacle is to provide a descriptive model of spectral data,
where each absorption or emission feature is characterized by an informative
Voigt profile. To that end, there are several ways in which users can generate
this model to more perfectly match their data, or the data they wish to create.

Defining output data
--------------------

Support for three different types of output data exists: `flux`,
`flux_decrement`, and `optical_depth`. This indicates the type of data that
will be outputted when the model is run. Output type can be specified upon
creation of a :class:`spectacle.modeling.models.Spectral1D` object::

    spec_mod = Spectral1D("HI1216", output='optical_depth')

Spectacle internally deals in optical depth space, and optical depth
information is transformed into flux as a step in the compound model.

For flux transformations:

.. math:: f(y) = np.exp(-y) - 1

And for flux decrement transformations:

.. math:: f(y) = 1 - np.exp(-y) - 1

All output types use the continuum information when depositing
absorption or emission data into the dispersion bins. Likewise, `flux` and
`flux_decrement` will generate results that may be saturated.

.. plot::
    :include-source:
    :align: center
    :context: close-figs

    >>> from astropy import units as u
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> from spectacle.modeling import Spectral1D, OpticalDepth1D

    >>> line = OpticalDepth1D("HI1216", v_doppler=50 * u.km/u.s, column_density=14)
    >>> line2 = OpticalDepth1D("HI1216", delta_v=100 * u.km/u.s)

    >>> spec_mod = Spectral1D([line, line2], continuum=1, output='flux')
    >>> x = np.linspace(-500, 500, 1000) * u.Unit('km/s')

    >>> flux = spec_mod(x)
    >>> flux_dec = spec_mod.as_flux_decrement(x)
    >>> tau = spec_mod.as_optical_depth(x)

    >>> f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    >>> ax1.set_title("Flux")
    >>> ax1.step(x, flux)
    >>> ax2.set_title("Flux Decrement")
    >>> ax2.step(x, flux_dec)
    >>> ax3.set_title("Optical Depth")
    >>> ax3.step(x, tau)
    >>> ax3.set_xlabel('Velocity [km/s]')
    >>> f.tight_layout()

Applying line spread functions
------------------------------

LSFs can be added to the :class:`spectacle.modeling.models.Spectral1D` model to
generate data that more appropriately matches what one might expect from an
instrument like, e.g., HST COS

.. plot::
    :include-source:
    :align: center
    :context: close-figs

    >>> from astropy import units as u
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> from spectacle.modeling import Spectral1D, OpticalDepth1D

    >>> line = OpticalDepth1D("HI1216", v_doppler=500 * u.km/u.s, column_density=14)
    >>> line2 = OpticalDepth1D("OVI1038", v_doppler=500 * u.km/u.s)

    >>> spec_mod = Spectral1D([line2], continuum=1, rest_wavelength=900 * u.AA, lsf='cos', output='flux')
    >>> x = np.linspace(0, 2000, 1000) * u.Unit('Angstrom')

    >>> f, ax = plt.subplots()
    >>> ax.set_title("Flux")
    >>> ax.step(x, spec_mod(x))

Converting dispersions
----------------------

Implementing redshift
---------------------