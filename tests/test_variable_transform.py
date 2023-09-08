import numpy as np

from virocon import (hs_s_to_hs_tz, hs_tz_to_hs_s, hs_tz_to_s_d, hs_tz_to_s_tz, s_tz_to_hs_tz, s_d_to_hs_tz)

def test_variable_transformations():
    n = 100
    hs = np.linspace(0.5, 10, num=n)
    tz = np.linspace(2, 15, num=n)

    hs_hat, tz_hat = s_d_to_hs_tz(*hs_tz_to_s_d(hs, tz))
    np.testing.assert_allclose(hs_hat, hs)
    np.testing.assert_allclose(tz_hat, tz)

    hs_hat, tz_hat = hs_tz_to_hs_s(*hs_s_to_hs_tz(hs, tz))
    np.testing.assert_allclose(hs_hat, hs)
    np.testing.assert_allclose(tz_hat, tz)

    hs_hat, tz_hat = hs_tz_to_s_tz(*s_tz_to_hs_tz(hs, tz))
    np.testing.assert_allclose(hs_hat, hs)
    np.testing.assert_allclose(tz_hat, tz)