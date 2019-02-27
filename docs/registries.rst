.. _registries:

Registries
==========

Spectacle uses an internal database of ions in order to look up relevant
atomic line information (specifically, oscillator strength, gamma values,
and rest-frame wavelength). Spectacle provides an extensive default ion registry
taken from Morton 2003. However, it is possible for users to provide their own
registries.

Spectacle searches the line registry for an atomic transition with the closest
rest-frame wavelength to the line in question. Alternatively, users can provide
a specific set of lines with associated atomic information for Spectacle to use.

.. code-block:: python

    >>> from spectacle.registries import line_registry
    >>> import astropy.units as u

Users can query the registry by passing in the restframe wavelength,
:math:`\lambda_0`, information for an ion

.. code-block:: python

    >>> from spectacle.registries import line_registry
    >>> line_registry.with_lambda(1216 * u.AA)
    <Row index=0>
     name     wave   osc_str    gamma
            Angstrom
     str9   float64  float64   float64
    ------ --------- ------- -----------
    HI1216 1215.6701  0.4164 626500000.0

Alternatively users can pass ion names in their queries. Spectacle will
attempt to find the closets alpha-numerical match using an internal
auto-correct:

.. code-block:: python

    >>> from spectacle.registries import line_registry
    >>> line_registry.with_name("HI1215")  # doctest: +SKIP
    spectacle [INFO    ]: Found line with name 'HI1216' from given name 'HI1215'.
    <Row index=0>
     name     wave   osc_str    gamma
            Angstrom
     str9   float64  float64   float64
    ------ --------- ------- -----------
    HI1216 1215.6701  0.4164 626500000.0


The default ion registry can be seen in its entirety
`here <https://github.com/MISTY-pipeline/spectacle/blob/master/spectacle/data/atoms.ecsv>`_.

Adding your own ion registry
----------------------------

Users can provide their own registries to replace the internal default ion
database used by Spectacle. The caveat is that the file must be an
`ECSV <http://docs.astropy.org/en/stable/api/astropy.io.ascii.Ecsv.html>`_ file
with four columns: ``name``, ``wave``, ``osc_str``, ``gamma``. The user's file
can then be loaded by importing and the :class:`spectacle.registries.lines.LineRegistry`
and declaring the `line_registry` variable

.. code-block:: python

    >>> from spectacle.registries import LineRegistry
    >>> line_registry = LineRegistry.read("/path/to/ion_file.ecsv")  # doctest: +SKIP