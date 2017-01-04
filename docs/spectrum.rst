

Creating and Interacting with Spectra
=====================================

The core spectrum object :py:class:`spectacle.core.spectra.Spectrum` is used to
contain all elements and behaviors of a spectrum. This includes the ability to
pass around and propagate uncertainties, convert between wavelength and
velocity space, calculate statistical values of absorption lines, and much
more.

.. note:: The uncertainty is considered to be the standard deviation, and all
          error propagation currently works under this assumption.


The :py:class:`spectacle.core.spectra.Spectrum` class