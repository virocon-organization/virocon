import pytest
import numpy as np
import matplotlib.pyplot as plt

from virocon import (
    read_ec_benchmark_dataset,
    IFORMContour,
    DependenceFunction,
    LogNormalDistribution,
    WeibullDistribution,
    GlobalHierarchicalModel,
    get_DNVGL_Hs_Tz,
    get_OMAE2020_V_Hs,
    plot_marginal_quantiles,
    plot_dependence_functions,
    plot_2D_isodensity,
    plot_2D_contour
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
    power3 = DependenceFunction(_power3, bounds, latex="$a + b * x^{c}$")
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


@pytest.fixture(scope="module")
def fitted_model():
    """
    Here we fit the joint distribution model described by Haselsteiner et al. (2020)
    to a dataset. We will use this model for various plot tests.
    """
    dist_descriptions, fit_descriptions, semantics = get_OMAE2020_V_Hs()
    model = GlobalHierarchicalModel(dist_descriptions)
    data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_D_1year.txt")
    model.fit(data, fit_descriptions)
    
    return model


@pytest.fixture(scope="module")
def semantics_fitted_model():
    """
    Semantics dictionary for the fitted_model.
    """
    dist_descriptions, fit_descriptions, semantics = get_OMAE2020_V_Hs()

    return semantics


def test_plot_dependence_function(seastate_model, fitted_model, semantics_fitted_model):
    plot_dependence_functions(seastate_model)
    semantics = {
        "names": ["Significant wave height", "Zero-up-crossing wave period"],
        "symbols": ["H_s", "T_z"],
        "units": ["m", "s"],
    }
    plot_dependence_functions(seastate_model, semantics)
    plot_dependence_functions(fitted_model, semantics_fitted_model)
    #plt.show()
