


def test_single_line():
    line1 = OpticalDepth1D(lambda_0=1216 * u.AA, v_doppler=10 * u.km / u.s,
                           column_density=13, delta_v=0 * u.km / u.s)

    spec_mod = Spectral1D([line1], z=0, continuum=0,
                          output='optical_depth')
