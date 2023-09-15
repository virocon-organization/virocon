"""
Common joint model structure are predefined in this module.
"""

import numpy as np

from virocon import (
    WeibullDistribution,
    LogNormalDistribution,
    ExponentiatedWeibullDistribution,
    DependenceFunction,
    WidthOfIntervalSlicer,
    variable_transform,
)

__all__ = [
    "get_DNVGL_Hs_Tz",
    "get_DNVGL_Hs_U",
    "get_OMAE2020_Hs_Tz",
    "get_OMAE2020_V_Hs",
    "get_Windmeier_EW_Hs_S",
    "get_Nonzero_EW_Hs_S",
]


def get_DNVGL_Hs_Tz():
    """
    Get DNVGL significant wave height and wave period model.

    Get the descriptions necessary to create th significant wave height
    and wave period model as defined in DNVGL [1]_ in section 3.6.3.

    Returns
    -------
    dist_descriptions : list of dict
        List of dictionaries containing the dist descriptions for each dimension.
        Can be used to create a GlobalHierarchicalModel.
    fit_descriptions : None
        Default fit is used so None is returned.
        Can be passed to fit function of GlobalHierarchicalModel.
    semantics : dict
        Dictionary with a semantic description of the model.
        Can be passed to plot functions.

    References
    ----------
    .. [1] DNV GL (2017). Recommended practice DNVGL-RP-C205: Environmental
        conditions and environmental loads.

    """

    # TODO docstrings with links to literature
    # DNVGL 3.6.3
    def _power3(x, a, b, c):
        return a + b * x**c

    def _exp3(x, a, b, c):
        return a + b * np.exp(c * x)

    bounds = [(0, None), (0, None), (None, None)]

    power3 = DependenceFunction(_power3, bounds, latex="$a + b * x^c$")
    exp3 = DependenceFunction(_exp3, bounds, latex="$a + b * \exp(c * x)$")

    dist_description_hs = {
        "distribution": WeibullDistribution(),
        "intervals": WidthOfIntervalSlicer(width=0.5),
    }

    dist_description_tz = {
        "distribution": LogNormalDistribution(),
        "conditional_on": 0,
        "parameters": {"mu": power3, "sigma": exp3},
    }

    dist_descriptions = [dist_description_hs, dist_description_tz]

    fit_descriptions = None

    semantics = {
        "names": ["Significant wave height", "Zero-up-crossing period"],
        "symbols": ["H_s", "T_z"],
        "units": ["m", "s"],
    }

    return dist_descriptions, fit_descriptions, semantics


def get_DNVGL_Hs_U():
    """
    Get DNVGL significant wave height and wind speed model.

    Get the descriptions necessary to create the significant wave height
    and wind speed model as defined in DNVGL [2]_ in section 3.6.4.

    Returns
    -------
    dist_descriptions : list of dict
        List of dictionaries containing the dist descriptions for each dimension.
        Can be used to create a GlobalHierarchicalModel.
    fit_descriptions : None
        Default fit is used so None is returned.
        Can be passed to fit function of GlobalHierarchicalModel.
    semantics : dict
        Dictionary with a semantic description of the model.
        Can be passed to plot functions.

    References
    ----------
    .. [2] DNV GL (2017). Recommended practice DNVGL-RP-C205: Environmental
        conditions and environmental loads.
    """

    def _power3(x, a, b, c):
        return a + b * x**c

    bounds = [(0, None), (0, None), (None, None)]

    alpha_dep = DependenceFunction(_power3, bounds=bounds, latex="$a + b * x^c$")
    beta_dep = DependenceFunction(_power3, bounds=bounds, latex="$a + b * x^c$")

    dist_description_hs = {
        "distribution": WeibullDistribution(),
        "intervals": WidthOfIntervalSlicer(width=0.5, min_n_points=20),
    }

    dist_description_u = {
        "distribution": WeibullDistribution(f_gamma=0),
        "conditional_on": 0,
        "parameters": {
            "alpha": alpha_dep,
            "beta": beta_dep,
        },
    }

    dist_descriptions = [dist_description_hs, dist_description_u]

    fit_descriptions = None

    semantics = {
        "names": ["Significant wave height", "Wind speed"],
        "symbols": ["H_s", "U"],
        "units": ["m", "m s$^{-1}$"],
    }

    return dist_descriptions, fit_descriptions, semantics


def get_OMAE2020_Hs_Tz():
    """
    Get OMAE2020 significant wave height and wave period model.

    Get the descriptions necessary to create the significant wave height
    and wave period model as described by Haselsteiner et al. [3]_.

    Returns
    -------
    dist_descriptions : list of dict
        List of dictionaries containing the dist descriptions for each dimension.
        Can be used to create a GlobalHierarchicalModel.
    fit_descriptions : list of dict
        List of dictionaries containing the fit description for each dimension.
        Can be passed to fit function of GlobalHierarchicalModel.
    semantics : dict
        Dictionary with a semantic description of the model.
        Can be passed to plot functions.

    References
    ----------
    .. [3] Haselsteiner, A.F.; Sander, A.; Ohlendorf, J.H.; Thoben, K.D. (2020)
        Global hierarchical models for wind and wave contours: Physical
        interpretations of the dependence functions. OMAE 2020, Fort Lauderdale,
        USA. Proceedings of the 39th International Conference on Ocean,
        Offshore and Arctic Engineering.
    """

    def _asymdecrease3(x, a, b, c):
        return a + b / (1 + c * x)

    def _lnsquare2(x, a, b, c):
        return np.log(a + b * np.sqrt(np.divide(x, 9.81)))

    bounds = [(0, None), (0, None), (None, None)]

    sigma_dep = DependenceFunction(
        _asymdecrease3, bounds=bounds, latex="$a + b / (1 + c * x)$"
    )
    mu_dep = DependenceFunction(
        _lnsquare2, bounds=bounds, latex="$\ln(a + b \sqrt{x / 9.81})$"
    )

    dist_description_hs = {
        "distribution": ExponentiatedWeibullDistribution(),
        "intervals": WidthOfIntervalSlicer(width=0.5, min_n_points=50),
    }

    dist_description_tz = {
        "distribution": LogNormalDistribution(),
        "conditional_on": 0,
        "parameters": {
            "sigma": sigma_dep,
            "mu": mu_dep,
        },
    }

    dist_descriptions = [dist_description_hs, dist_description_tz]

    fit_description_hs = {"method": "wlsq", "weights": "quadratic"}
    fit_descriptions = [fit_description_hs, None]

    semantics = {
        "names": ["Significant wave height", "Zero-up-crossing period"],
        "symbols": ["H_s", "T_z"],
        "units": ["m", "s"],
    }

    return dist_descriptions, fit_descriptions, semantics


def get_OMAE2020_V_Hs():
    """
    Get OMAE2020 wind speed and significant wave height model.

    Get the descriptions necessary to create the wind speed and
    significant wave height model as described by Haselsteiner et al. [4]_.

    Returns
    -------
    dist_descriptions : list of dict
        List of dictionaries containing the dist descriptions for each dimension.
        Can be used to create a GlobalHierarchicalModel.
    fit_descriptions : list of dict
        List of dictionaries containing the fit description for each dimension.
        Can be passed to fit function of GlobalHierarchicalModel.
    semantics : dict
        Dictionary with a semantic description of the model.
        Can be passed to plot functions.

    References
    ----------
    .. [4] Haselsteiner, A.F.; Sander, A.; Ohlendorf, J.H.; Thoben, K.D. (2020)
        Global hierarchical models for wind and wave contours: Physical
        interpretations of the dependence functions. OMAE 2020, Fort Lauderdale,
        USA. Proceedings of the 39th International Conference on Ocean,
        Offshore and Arctic Engineering.
    """

    def _logistics4(x, a=1, b=1, c=-1, d=1):
        return a + b / (1 + np.exp(c * (x - d)))

    def _alpha3(x, a, b, c, d_of_x):
        return (a + b * x**c) / 2.0445 ** (1 / d_of_x(x))

    logistics_bounds = [(0, None), (0, None), (None, 0), (0, None)]

    alpha_bounds = [(0, None), (0, None), (None, None)]

    beta_dep = DependenceFunction(
        _logistics4,
        logistics_bounds,
        weights=lambda x, y: y,
        latex="$a + b / (1 + \exp[c * (x -d)])$",
    )
    alpha_dep = DependenceFunction(
        _alpha3,
        alpha_bounds,
        d_of_x=beta_dep,
        weights=lambda x, y: y,
        latex="$(a + b * x^c) / 2.0445^{1 / F()}$",
    )

    dist_description_v = {
        "distribution": ExponentiatedWeibullDistribution(),
        "intervals": WidthOfIntervalSlicer(2, min_n_points=50),
    }

    dist_description_hs = {
        "distribution": ExponentiatedWeibullDistribution(f_delta=5),
        "conditional_on": 0,
        "parameters": {
            "alpha": alpha_dep,
            "beta": beta_dep,
        },
    }

    dist_descriptions = [dist_description_v, dist_description_hs]

    fit_description_v = {"method": "wlsq", "weights": "quadratic"}
    fit_description_hs = {"method": "wlsq", "weights": "quadratic"}
    fit_descriptions = [fit_description_v, fit_description_hs]

    semantics = {
        "names": ["Wind speed", "Significant wave height"],
        "symbols": ["V", "H_s"],
        "units": [
            "m s$^{-1}$",
            "m",
        ],
    }

    return dist_descriptions, fit_descriptions, semantics


def get_Windmeier_EW_Hs_S():
    """
    Get Windmeier's EW sea state model.

    Get the descriptions necessary to create the significant wave height - steepness
    model that was proposed by Windmeier [5]_. Both, Hs and Steepness follow
    an exponentiated Weibull distribution.

    Because the model is defined in Hs-steepness space it must be transformed to
    Hs-Tz for contour calculation.

    Returns
    -------
    dist_descriptions : list of dict
        List of dictionaries containing the dist descriptions for each dimension.
        Can be used to create a GlobalHierarchicalModel.
    fit_descriptions : None
        Default fit is used so None is returned.
        Can be passed to fit function of GlobalHierarchicalModel.
    semantics : dict
        Dictionary with a semantic description of the model.
        Can be passed to plot functions.
    transformations : dict

    References
    ----------
    .. [5] Windmeier, K.-L. (2022). Modeling the statistical distribution of sea state
    parameters [Master Thesis, University of Bremen]. https://doi.org/10.26092/elib/2181
    """

    def _linear2(x, a=0, b=1):
        return a + b * x

    def _limited_growth2(x, a=0.08, b=1):
        return a * (1 - np.exp(-b * x))

    def _transform(hs_tz):
        hs = hs_tz[:, 0]
        tz = hs_tz[:, 1]
        s, _ = variable_transform.hs_tz_to_s_d(hs, tz)
        return np.c_[hs, s]

    def _inv_transform(hs_s):
        hs = hs_s[:, 0]
        s = hs_s[:, 1]
        hs, tz = variable_transform.hs_s_to_hs_tz(hs, s)
        return np.c_[hs, tz]

    def _jacobian(hs_s):
        hs = hs_s[:, 0]
        s = hs_s[:, 1]
        return 2 * variable_transform.factor * hs / s**3

    linear_2_bounds = [(0, None), (0, None)]
    limited_growth2_bounds = [(0, 1), (0, None)]

    linear2 = DependenceFunction(_linear2, bounds=linear_2_bounds)
    limited_growth2 = DependenceFunction(
        _limited_growth2, bounds=limited_growth2_bounds
    )

    dist_description_hs = {
        "distribution": ExponentiatedWeibullDistribution(),
        "intervals": WidthOfIntervalSlicer(width=0.5, min_n_points=50),
    }

    dist_description_s = {
        "distribution": ExponentiatedWeibullDistribution(f_delta=2.35),
        "conditional_on": 0,
        "parameters": {"alpha": limited_growth2, "beta": linear2},
    }

    dist_descriptions = [dist_description_hs, dist_description_s]

    fit_description_hs = {"method": "wlsq", "weights": "quadratic"}
    fit_descriptions = [fit_description_hs, None]

    transformations = transformations = {
        "transform": _transform,
        "inverse": _inv_transform,
        "jacobian": _jacobian,
    }

    semantics = {
        "names": ["Significant wave height", "Zero-up-crossing period"],
        "symbols": ["H_s", "T_z"],
        "units": ["m", "s"],
    }

    return dist_descriptions, fit_descriptions, semantics, transformations


def get_Nonzero_EW_Hs_S():
    """
    Get the non-zero EW sea state model.

    Get the descriptions necessary to create the significant wave height - steepness
    model that is an adaptation of Windmeier's EW model [5]_. Both, Hs and Steepness follow
    an exponentiated Weibull distribution.

    Compared to Windmeier's EW model, this model has a dependence function for scale
    that evaluates to scale > 0 at hs = 0 m .
    The dependence function reads: 0.005 + a * (1 - np.exp(-b * hs))

    Because the model is defined in Hs-steepness space it must be transformed to
    Hs-Tz for contour calculation.

    Returns
    -------
    dist_descriptions : list of dict
        List of dictionaries containing the dist descriptions for each dimension.
        Can be used to create a GlobalHierarchicalModel.
    fit_descriptions : None
        Default fit is used so None is returned.
        Can be passed to fit function of GlobalHierarchicalModel.
    semantics : dict
        Dictionary with a semantic description of the model.
        Can be passed to plot functions.
    transformations : dict

    References
    ----------
    .. [5] Windmeier, K.-L. (2022). Modeling the statistical distribution of sea state
    parameters [Master Thesis, University of Bremen]. https://doi.org/10.26092/elib/2181
    """

    def _linear2(x, a=0, b=1):
        return a + b * x

    def _limited_growth2(x, a=0.08, b=1):
        # Idea to start at > 0 is based on the Figure 4.10 in Windmeier's thesis.
        return 0.005 + a * (1 - np.exp(-b * x))

    def _transform(hs_tz):
        hs = hs_tz[:, 0]
        tz = hs_tz[:, 1]
        s, _ = variable_transform.hs_tz_to_s_d(hs, tz)
        return np.c_[hs, s]

    def _inv_transform(hs_s):
        hs = hs_s[:, 0]
        s = hs_s[:, 1]
        hs, tz = variable_transform.hs_s_to_hs_tz(hs, s)
        return np.c_[hs, tz]

    def _jacobian(hs_s):
        hs = hs_s[:, 0]
        s = hs_s[:, 1]
        return 2 * variable_transform.factor * hs / s**3

    linear_2_bounds = [(0, None), (0, None)]
    limited_growth2_bounds = [(0, 1), (0, None)]

    linear2 = DependenceFunction(_linear2, bounds=linear_2_bounds)
    limited_growth2 = DependenceFunction(
        _limited_growth2, bounds=limited_growth2_bounds
    )

    dist_description_hs = {
        "distribution": ExponentiatedWeibullDistribution(),
        "intervals": WidthOfIntervalSlicer(width=0.5, min_n_points=50),
    }

    dist_description_s = {
        "distribution": ExponentiatedWeibullDistribution(f_delta=2.35),
        "conditional_on": 0,
        "parameters": {"alpha": limited_growth2, "beta": linear2},
    }

    dist_descriptions = [dist_description_hs, dist_description_s]

    fit_description_hs = {"method": "wlsq", "weights": "quadratic"}
    fit_descriptions = [fit_description_hs, None]

    transformations = transformations = {
        "transform": _transform,
        "inverse": _inv_transform,
        "jacobian": _jacobian,
    }

    semantics = {
        "names": ["Significant wave height", "Zero-up-crossing period"],
        "symbols": ["H_s", "T_z"],
        "units": ["m", "s"],
    }

    return dist_descriptions, fit_descriptions, semantics, transformations
