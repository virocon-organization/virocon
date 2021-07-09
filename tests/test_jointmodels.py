import pytest
import numpy as np

from virocon import (
    DependenceFunction,
    LogNormalDistribution,
    WeibullDistribution,
    GlobalHierarchicalModel,
)


@pytest.fixture(scope="module")
def seastate_model():
    """
    This joint distribution model described by Vanem and Bitner-Gregersen (2012)
    is widely used in academia. Here, we use it for evaluation.
    DOI: 10.1016/j.apor.2012.05.006
    """

    def _power3(x, a=0.1000, b=1.489, c=0.1901):
        return a + b * x ** c

    # A 3-parameter exponential function (a dependence function).
    def _exp3(x, a=0.0400, b=0.1748, c=-0.2243):
        return a + b * np.exp(c * x)

    bounds = [(0, None), (0, None), (None, None)]
    power3 = DependenceFunction(_power3, bounds)
    exp3 = DependenceFunction(_exp3, bounds)

    dist_description_0 = {
        "distribution": WeibullDistribution(alpha=2.776, beta=1.471, gamma=0.8888),
    }
    dist_description_1 = {
        "distribution": LogNormalDistribution(),
        "conditional_on": 0,
        "parameters": {"mu": power3, "sigma": exp3},
    }
    model = GlobalHierarchicalModel([dist_description_0, dist_description_1])

    return model


def test_joint_pdf(seastate_model):
    x = [2.5, 6]  # (hs, tz) in (m, s).
    f = seastate_model.pdf(x)

    # Second sea state's period is unphysical due to wave breaking.
    xs = [
        [2.5, 6],
        [2.5, 2],
    ]
    fs = seastate_model.pdf(xs)
    assert fs[0] > fs[1]


def test_joint_cdf(seastate_model):
    x = [2.5, 6]  # (hs, tz) in (m, s).
    F = seastate_model.cdf(x)
    xs = [[2.5, 2], [2.5, 6], [2.5, 10]]
    Fs = seastate_model.cdf(xs)
    assert Fs[1] > Fs[0]
    assert Fs[2] > Fs[1]
