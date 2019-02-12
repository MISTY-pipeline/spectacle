import pytest
from ..modeling import Spectral1D, OpticalDepth1D
import numpy as np
import astropy.units as u


def test_create_model():
    line1 = OpticalDepth1D(lambda_0=1216 * u.AA)
    line2 = OpticalDepth1D(name="HI1216")

    # Create model from lambda
    spec_mod_lambda = Spectral1D(line1)

    # Create model from name
    spec_mod_name = Spectral1D(line2)

    x = np.linspace(-100, 100, 200) * u.AA

    assert np.allclose(spec_mod_lambda(x), spec_mod_name(x))


def test_add_line():
    line1 = OpticalDepth1D(lambda_0=1216 * u.AA)
    line2 = OpticalDepth1D(name="HI1216")

    # Add line from pre-made instance
    spec_mod = Spectral1D(line1)
    spec_mod = spec_mod.with_line(line2)

    x = np.linspace(-100, 100, 200) * u.AA

    assert len(spec_mod(x)) == 200
    assert len(spec_mod.lines) == 2

    # Add line from name
    spec_mod = spec_mod.with_line("HI1216")

    assert len(spec_mod(x)) == 200
    assert len(spec_mod.lines) == 3