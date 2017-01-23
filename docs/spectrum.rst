

Handling Spectra
================

The core spectrum object :py:class:`spectacle.core.spectra.Spectrum` is used to
contain all elements and behaviors of a spectrum. This includes the ability to
pass around and propagate uncertainties, convert between wavelength and
velocity space, calculate statistical values of absorption lines, and much
more.

The :py:class:`spectacle.core.spectra.Spectrum` class inherits from the
:py:class:`astropy.nddata.NDDataRef` class, allowing for a unified IO, unit,
uncertainty, wcs, and mask interface.

.. note:: The uncertainty is considered to be the standard deviation, and all
error propagation currently works under this assumption.


Creating Spectra
----------------

Spectra objects can be created one of two ways: either defining their data and
associated properties manually, or by reading in a recognized file.

Reading in a file requires that IO registry contains some way of recognizing
the file format. Basic formats, as well as how to include custom formats, can
be seen in :doc:`Reading and Writing Files </reading_writing.rst>`.

Once the registry is aware of the format of the file, creating a spectrum
object is as simple as::

    >>> spectrum = Spectrum1D.read("my_spectrum_file.fits", format="my_format")

The format is defined by the custom IO registry you create.

To manually create the spectrum object is a matter of passing in the relevant
information to the spectrum class::

    >>> spectrum = Spectrum1D(flux_arr, dispersion=wavelength_arr,
                              disp_unit=wavelength_unit,
                              uncertainty=uncert_array)


Modeling Spectra
----------------

Spectral models are the de-facto means of either analytically creating a
spectrum, or quantifying already existing spectrum data.

The modeling system follows the Astropy approach, and as such, the spectrum
model is implemented as a :py:class:`spectacle.modeling.models.Absorption1D`
callable that returns a new spectrum object.

    >>> model = Absorption1D()
    >>> dispersion = np.arange(100)
    >>> spectrum = model(dispersion)

Lines are added to the model, not the spectrum object, as the latter is only a
container for spectral information.

    >>> model.add_line(name="NV1239", v_doppler=2.5e7, column_density=10**14.66)
    >>> spectrum_with_line = model(dispersion)

To get more information on the default line lists, or how to add custom line
lists, see :doc:`Line Lists </line_lists.rst>`