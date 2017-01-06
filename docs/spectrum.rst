

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
the file format. Basic formats, as well as how to include custom formats, have
can be seen in :doc:`Reading and Writing Files </reading_write.rst>`.

Once the registry is aware of the format of the file, creating a spectrum
object is as simple as::

    >>> spectrum = Spectrum1D.read("my_spectrum_file.fits")

To manually create the spectrum object is a matter of passing in the relevant
information to the spectrum class::

    >>> spectrum = Spectrum1D(flux_arr, dispersion=wavelength_arr, uncertainty=uncert_array)


Spectrum Models
---------------

The spectrum object also contains information about the best.