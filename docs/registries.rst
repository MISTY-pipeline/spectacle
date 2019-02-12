Registries
==========

Spectacle uses an internal database of ions in order to look up information
like the oscillator strength, gamma values, and centroid on lines. The default
ion registry is extensive and loaded by default. However, it is possible for
users to provide their own registries.

Registries are searched for the closest centroid information when used in
line fitting, or are directly queried for named ions in the case where users
provide line name information.

.. code-block:: python

    >>> from spectacle.registries import line_registry
    >>> import astropy.units as u

Users can query the registry by passing in :math:`\lambda_0` information
for an ion

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