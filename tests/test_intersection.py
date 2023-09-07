import numpy as np

from virocon._intersection import intersection


def test_intersection():
    x = np.linspace(1, 5, 11)
    y1 = 0 * x + 1
    y2 = 0.5 * x

    # The curves intersect at point (2, 1).
    ix, iy = intersection(x, y1, x, y2)

    np.testing.assert_allclose(ix, 2, atol=0.001)
    np.testing.assert_allclose(iy, 1, atol=0.001)
