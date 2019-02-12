Analysis
========

Spectacle comes with several statistics on the line profiles of models. These
can be accessed via the :meth:`~spectacle.modeling.models.Spectral1D.stats`
method on the spectrum object, and will display basic parameter information
such as the centroid, column density, dopper velocity, etc of each line in
the spectrum. Three additional statistics are added to this table as well: the
equivalent width, the velocity delta covering 90% of the flux value, and
the full width half maximum of the feature.

.. code-block:: python

    spec_mod.stats(x)

    <QTable length=2>
      name     wave   col_dens  v_dop  delta_v          ew                dv90                fwhm
             Angstrom           km / s  km / s       Angstrom            km / s             Angstrom
    bytes10  float64  float64  float64 float64       float64            float64             float64
    ------- --------- -------- ------- ------- ------------------- ------------------ -------------------
     HI1216 1215.6701     14.0    10.0     0.0  0.0816268618521417 18.500568226928248 0.08948166854406736
    OVI1038 1037.6167     14.0    70.0     0.0 0.43715323945673995 151.75522375824642 0.40864380730636185

