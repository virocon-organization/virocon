"""
Functions to plot distributions and contours.
"""

import re
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts

from matplotlib.colors import LinearSegmentedColormap

from virocon.utils import calculate_design_conditions

__all__ = [
    "plot_marginal_quantiles",
    "plot_dependence_functions",
    "plot_histograms_of_interval_distributions",
    "plot_2D_isodensity",
    "plot_2D_contour",
]


# colors (schemes) chosen according to https://personal.sron.nl/~pault/


def _rainbow_PuRd():
    """
    Thanks to Paul Tol (https://personal.sron.nl/~pault/data/tol_colors.py)
    License:  Standard 3-clause BSD
    """
    clrs = [
        "#6F4C9B",
        "#6059A9",
        "#5568B8",
        "#4E79C5",
        "#4D8AC6",
        "#4E96BC",
        "#549EB3",
        "#59A5A9",
        "#60AB9E",
        "#69B190",
        "#77B77D",
        "#8CBC68",
        "#A6BE54",
        "#BEBC48",
        "#D1B541",
        "#DDAA3C",
        "#E49C39",
        "#E78C35",
        "#E67932",
        "#E4632D",
        "#DF4828",
        "#DA2222",
    ]
    cmap = LinearSegmentedColormap.from_list("rainbow_PuRd", clrs)
    cmap.set_bad("#FFFFFF")
    return cmap


# TODO move to utility as it is also used in contours.py
def get_default_semantics(n_dim):
    """
    Generate default semantics for n_dim dimensions.

    Parameters
    ----------
    n_dim : int
        Number of dimensions. Indicating the number of variables of the model.

    Returns
    -------
    semantics: dict
        Generated model description.

    """

    semantics = {
        "names": [f"Variable {dim + 1}" for dim in range(n_dim)],
        "symbols": [f"X_{dim + 1}" for dim in range(n_dim)],
        "units": ["arb. unit" for dim in range(n_dim)],
    }
    return semantics


def _get_n_axes(n_intervals):
    if n_intervals > 9:
        raise NotImplementedError(
            "Automatic axes creation is only supported for up to 9 intervals."
        )

    table = [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 2),
        (2, 3),
        (2, 3),
        (3, 3),
        (3, 3),
        (3, 3),
    ]

    fig, axes = plt.subplots(
        *table[n_intervals], sharex=True, sharey=True, squeeze=False
    )
    return fig, axes.ravel()


def plot_marginal_quantiles(model, sample, semantics=None, axes=None):
    """
    Plot all marginal quantiles of a model.

    Plots the fitted marginal distribution versus a dataset in a quantile-quantile (QQ) plot.

    Parameters
    ----------
    model :  MultivariateModel
        The model used to plot the marginal quantiles.
    sample : ndarray of floats
        The environmental data sample that should be plotted against the fit.
        Shape: (sample_size, n_dim)
    semantics: dict, optional
        The description of the model. If None (the default) generic semantics
        will be used. The structure is as follows:
        modeldesc = {
        "names" : [<Names of variables>],
        "symbols" : [<Description of symbols>],
        "units" : [<Units of variables>] }
    axes: list, optional
        The matplotlib axes objects to plot into. One for each dimension. If
        None (the default) a new figure will be created for each dimension.

    Returns
    -------
    The used matplotlib axes object.

    Notes
    -----
    When saving the resulting axes in a vector image format (pdf, svg) the
    `sample` will still be rasterized to reduce the file size.
    To prevent that, iterate over the axes and use
    ``ax.get_lines()[0].set_rasterized(False)``.

    """

    sample = np.asarray(sample)
    n_dim = model.n_dim
    if semantics is None:
        semantics = get_default_semantics(n_dim)

    if axes is None:
        axes = []
        for i in range(n_dim):
            _, ax = plt.subplots()
            axes.append(ax)

    # probplot expects an object that has a ppf method, but we name it icdf
    # therefor we create a wrapper that maps the ppf to the icdf method
    class MarginalDistWrapper:
        def __init__(self, model, idx):
            self.model = model
            self.idx = idx

        def ppf(self, q):
            return self.model.marginal_icdf(q, self.idx)

    for dim in range(n_dim):
        dist_wrapper = MarginalDistWrapper(model, dim)
        ax = axes[dim]

        sts.probplot(sample[:, dim], dist=dist_wrapper, fit=False, plot=ax)
        ax.get_lines()[0].set_markerfacecolor("k")
        ax.get_lines()[0].set_markeredgecolor("k")
        ax.get_lines()[0].set_marker("x")
        ax.get_lines()[0].set_markersize(3)

        # Because a sample usually holds much more than 1000 observations, the
        # output shall be rasterized to reduce file size if the figure is saved
        # in a vector file format (svg, pdf).
        ax.get_lines()[0].set_rasterized(True)

        # prior to scipy version 1.7.0 a regression fit line was plotted,
        # even with option fit=False. As discussed in PR#149 we do not want
        # to keep this line. So here we remove it if it was plotted.
        if len(ax.lines) > 1:
            ax.lines[1].remove()

        # draw 45° line
        # adapted from statsmodels.graphics.gofplots.qqline
        # https://github.com/statsmodels/statsmodels/blob/161ca84b2b6e2b3ba5ef6b570d8c907c9b06f5de/statsmodels/graphics/gofplots.py#L892-L897
        end_pts = list(zip(ax.get_xlim(), ax.get_ylim()))
        end_pts[0] = min(end_pts[0])
        end_pts[1] = max(end_pts[1])
        ax.plot(end_pts, end_pts, c="#BB5566")
        ax.set_xlim(end_pts)
        ax.set_ylim(end_pts)

        name_and_unit = f"{semantics['names'][dim].lower()} ({semantics['units'][dim]})"
        ax.set_xlabel(f"Theoretical quantiles of {name_and_unit}")
        ax.set_ylabel(f"Ordered values of {name_and_unit}")
        ax.title.set_text("")

    return axes


def plot_dependence_functions(model, semantics=None, par_rename={}, axes=None):
    """
    Plot the fitted dependence functions of a model.

    Parameters
    ----------
    model :  MultivariateModel
        The model with the fitted dependence functions.
    semantics: dict, optional
        The description of the model. If None (the default) generic semantics will be used.
    par_rename : dict, optional
        A dictionary that maps from names of conditional parameters to a
        string. If e.g. the model has a distribution with a conditional
        parameter named 'mu' one could change that in the plot to '$mu$' with
        {'mu': '$mu$'}.
    axes: list of matplotlib axes objects, optional
        If not further specified, the number of axes are dependent on the number
        of dimensions of the model. Defaults to None.

    Returns
    -------
    The used matplotlib axes object.

    """

    n_dim = model.n_dim
    conditional_dist_idc = [
        dim for dim in range(n_dim) if model.conditional_on[dim] is not None
    ]

    if semantics is None:
        semantics = get_default_semantics(n_dim)

    if axes is None:
        n_axes = 0
        for dim in conditional_dist_idc:
            n_axes += len(model.distributions[dim].conditional_parameters)
        axes = []
        for i in range(n_axes):
            _, ax = plt.subplots()
            axes.append(ax)

    axes_counter = 0
    for dim in conditional_dist_idc:
        dist = model.distributions[dim]
        conditioning_values = dist.conditioning_values
        if conditioning_values is not None:
            x = np.linspace(0, max(conditioning_values))
        else:
            x = np.linspace(0, 10)
        cond_idx = model.conditional_on[dim]
        x_name = semantics["names"][cond_idx]
        x_symbol = semantics["symbols"][cond_idx]
        x_unit = semantics["units"][cond_idx]
        x_label = f"{x_name}," + r" $\it{" + f"{x_symbol}" + r"}$" + f" ({x_unit})"
        for par_name, dep_func in dist.conditional_parameters.items():
            ax = axes[axes_counter]
            axes_counter += 1
            # If a model is created directly (without fitting it to a dataset), it does
            # not have conditioning values.
            if conditioning_values is not None:
                par_values = [par[par_name] for par in dist.parameters_per_interval]
                ax.scatter(
                    conditioning_values,
                    par_values,
                    c="k",
                    marker="x",
                    label="estimates from intervals",
                )
            if dep_func.latex is not None:
                dep_func_label = dep_func.latex

                # Replace 'x' with variable symbol (e.g. 'h_s')
                splitted_symbol = x_symbol.split("_")
                if len(splitted_symbol) == 1:  # If there was no underscore.
                    var_symbol = splitted_symbol[0].lower()
                else:  # If there was one or many underscores.
                    var_symbol = (
                            splitted_symbol[0].lower() + "_" + "_".join(splitted_symbol[1:])
                    )

                # Replace 'x' if it is not part of '\exp' which is checked by checking whether
                # 'x' follows '\e'.
                dep_func_label = re.sub(
                    r"(?<!\\e)x", "{" + var_symbol + "}", dep_func_label
                )

                # Replace symbols of parameters (a, b, ..) with estimated values.
                for par_name_local, par_value_local in dep_func.parameters.items():
                    dep_func_label = dep_func_label.replace(
                        par_name_local, "{" + "{:.2f}".format(par_value_local) + "}"
                    )
            else:
                if not isinstance(dep_func.func, partial):
                    dep_func_label = "Dependence function: " + dep_func.func.__name__
                else:
                    dep_func_label = (
                            "Dependence function: " + dep_func.func.func.__name__
                    )
            ax.plot(x, dep_func(x), c="#004488", label=dep_func_label)
            ax.set_xlabel(x_label)
            if par_name in par_rename:
                y_label = par_rename[par_name]
            else:
                y_label = par_name
            ax.set_ylabel(y_label)
            ax.legend()

    return axes


def plot_histograms_of_interval_distributions(model, sample, semantics=None, plot_pdf=True):
    """
    Plot histograms of all model dimensions.

    If the model is conditional in a dimension all histograms per interval are plotted for that dimension.
    In such a case the fitted interval distributions are generally different from the final joint distribution.

    Parameters
    ----------
    model : MultivariateModel
        The model that was fitted to the dataset.
    sample : ndarray
        The data that was used to fit the model.
    semantics: dict, optional
        The description of the model. If None (the default) generic semantics will be used.
    plot_pdf: boolean, optional
        Whether the fitted probability density should be plotted. Defaults to True.

    Returns
    -------
    The used matplotlib axes objects.

    """
    sample = np.asarray(sample)
    n_dim = model.n_dim

    if semantics is None:
        semantics = get_default_semantics(n_dim)

    figures = []
    axes_list = []

    for dim in range(n_dim):

        x_name = semantics["names"][dim]
        x_symbol = semantics["symbols"][dim]
        x_unit = semantics["units"][dim]
        x_label = f"{x_name}," + r" $\it{" + f"{x_symbol}" + r"}$" + f" ({x_unit})"

        if model.conditional_on[dim] is None:
            # unconditional
            data = sample[:, dim]
            dist = model.distributions[dim]
            x = np.linspace(np.min(data), np.max(data))
            fig, ax = plt.subplots()
            ax.hist(
                data,
                bins="doane",
                density=True,
                color="#000000",
                histtype="stepfilled",
                alpha=0.2,
            )
            if plot_pdf:
                density = dist.pdf(x)
                ax.plot(
                    x,
                    density,
                    color="#004488",
                )
            ax.set_xlabel(x_label)
            ax.set_ylabel("Probability density")
            ax.set_title(f"n={len(data)}")

            figures.append(fig)
            axes_list.append(ax)

        else:
            # conditional
            cond_dist = model.distributions[dim]
            dist_per_interval = cond_dist.distributions_per_interval
            n_intervals = len(dist_per_interval)

            conditioning_idx = model.conditional_on[dim]
            conditioning_symbol = semantics["symbols"][conditioning_idx]
            conditioning_unit = semantics["units"][conditioning_idx]

            slicer = model.interval_slicers[conditioning_idx]
            interval_slices, conditioning_values, _ = slicer.slice_(
                sample[:, conditioning_idx]
            )
            data_intervals = [sample[int_slice, dim] for int_slice in interval_slices]

            # if the following fails, the sample was probably not used for fitting
            # TODO raise proper exception then
            np.testing.assert_allclose(
                conditioning_values, cond_dist.conditioning_values
            )
            assert len(data_intervals) == len(cond_dist.data_intervals)
            for i in range(len(data_intervals)):
                np.testing.assert_allclose(
                    np.sort(data_intervals[i]), np.sort(cond_dist.data_intervals[i])
                )

            fig, axes = _get_n_axes(n_intervals)
            for interval_idx in range(n_intervals):
                cond_val = conditioning_values[interval_idx]
                data = data_intervals[interval_idx]
                x = np.linspace(np.min(data), np.max(data))
                dist = dist_per_interval[interval_idx]
                ax = axes[interval_idx]
                ax.hist(
                    data,
                    bins="doane",
                    density=True,
                    color="#000000",
                    histtype="stepfilled",
                    alpha=0.2,
                )
                if plot_pdf:
                    density = dist.pdf(x)
                    ax.plot(
                        x,
                        density,
                        color="#004488",
                    )
                ax.set_xlabel(x_label)
                ax.set_ylabel("Probability density")
                title = f"{conditioning_symbol} = {cond_val:.3f} {conditioning_unit}, n={len(data)}"
                ax.set_title(title)

            figures.append(fig)
            axes_list.append(axes)

    return figures, axes_list


def plot_2D_isodensity(
        model,
        sample,
        semantics=None,
        swap_axis=False,
        limits=None,
        levels=None,
        ax=None,
        n_grid_steps=250,
):
    """
    Plot isodensity contours and a data sample for a 2D model.

    Parameters
    ----------
    model :  MultivariateModel
        The 2D model to use.
    sample : ndarray of floats
        The 2D data sample that should be plotted.
    semantics: dict, optional
        The description of the model. If None (the default) generic semantics
        will be used.
    swap_axis : boolean, optional
        If True the second dimension of the model is plotted on the x-axis and
        the first on the y-axis. Otherwise, vice-versa. Defaults to False.
    limits : list of tuples, optional
        Specifies in which rectangular region the density is calculated and
        plotted. If None (default) limits will be set automatically.
        Example: [(0, 20), (0, 12)]
    levels : list of floats, optional
        The probability density levels that are plotted. If None (default)
        levels are set automatically.
        Example: [0.001, 0.01, 0.1]
    n_grid_steps : int, optional
        The number of steps along each axis of the grid used to plot the contours.
        Defaults to 250.
    ax : matplotlib Axes, optional
        Matplotlib axes object to use for plotting. If None (default) a new
        figure will be created.

    Returns
    -------
    The used matplotlib axes object.

    """

    n_dim = model.n_dim
    assert n_dim == 2

    if swap_axis:
        x_idx = 1
        y_idx = 0
    else:
        x_idx = 0
        y_idx = 1

    if semantics is None:
        semantics = get_default_semantics(n_dim)

    if ax is None:
        _, ax = plt.subplots()

    sample = np.asarray(sample)
    ax.scatter(
        sample[:, x_idx],
        sample[:, y_idx],
        c="k",
        marker=".",
        alpha=0.3,
        rasterized=True,
    )

    if limits is not None:
        x_lower = limits[0][0]
        x_upper = limits[0][1]
        y_lower = limits[1][0]
        y_upper = limits[1][1]
    else:
        x_range = max(sample[:, 0]) - min((sample[:, 0]))
        expand_factor = 0.05
        x_lower = min(sample[:, 0]) - expand_factor * x_range
        x_upper = max(sample[:, 0]) + expand_factor * x_range
        y_range = max(sample[:, 1]) - min((sample[:, 1]))
        y_lower = min(sample[:, 1]) - expand_factor * y_range
        y_upper = max(sample[:, 1]) + expand_factor * y_range

    x, y = np.linspace((x_lower, y_lower), (x_upper, y_upper), num=n_grid_steps).T
    X, Y = np.meshgrid(x, y)
    grid_flat = np.c_[X.ravel(), Y.ravel()]
    f = model.pdf(grid_flat)
    Z = f.reshape(X.shape)

    if swap_axis:
        tmp = X
        X = Y
        Y = tmp

    if levels is not None:
        lvl_labels = ["{:.1E}".format(x) for x in levels]
        n_levels = len(levels)
    else:
        # Define the lowest isodensity level based on the density values on the evaluated grid.
        q = np.quantile(f, q=0.5)
        if q > 0:
            min_lvl = int(f"{q:.0e}".split("e")[1])
        else:
            min_lvl = -5
        n_levels = np.abs(min_lvl)
        levels = np.logspace(-1, min_lvl, num=n_levels)[::-1]
        lvl_labels = [f"1E{int(i)}" for i in np.linspace(-1, min_lvl, num=n_levels)][
                     ::-1
                     ]

    cmap = _rainbow_PuRd()
    colors = cmap(np.linspace(0, 1, num=n_levels))
    CS = ax.contour(X, Y, Z, levels=levels, colors=colors)
    proxies = [plt.Line2D([], [], color=pc.get_edgecolor()[0]) for pc in CS.collections]
    ax.legend(
        proxies,
        lvl_labels,
        loc="upper left",
        ncol=1,
        frameon=False,
        title="Probability density",
    )
    x_name = semantics["names"][x_idx]
    x_symbol = semantics["symbols"][x_idx]
    x_unit = semantics["units"][x_idx]
    x_label = f"{x_name}," + r" $\it{" + f"{x_symbol}" + r"}$" + f" ({x_unit})"
    y_name = semantics["names"][y_idx]
    y_symbol = semantics["symbols"][y_idx]
    y_unit = semantics["units"][y_idx]
    y_label = f"{y_name}," + r" $\it{" + f"{y_symbol}" + r"}$" + f" ({y_unit})"
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    return ax


def plot_2D_contour(
        contour,
        sample=None,
        design_conditions=None,
        semantics=None,
        swap_axis=False,
        ax=None,
):
    """
    Plot a 2D contour.

    Parameters
    ----------
    contour: Contour
        The 2D contour that should be plotted.
    sample : ndarray of floats, optional
        A 2D data sample that should be plotted along the contour.
        Shape: (number of realizations, 2)
    design_conditions : array-like or boolean, optional
       Specified environmental conditions under which the system must operate.
       If an array it is assumed to be a 2D array of shape
       (number of points, 2) and should contain the precalculated design
       conditions. If it is True design_conditions are computed with default
       values and plotted. Otherwise, no design conditions will be plotted
       (the default).
    semantics: dict, optional
        Generated model description. Defaults to None.
    swap_axis : boolean, optional
        f True the second dimension of the model is plotted on the x-axis and
        the first on the y-axis. Otherwise, vice-versa. Defaults to False.
    ax : matplotlib Axes, optional
        Matplotlib axes object to use for plotting. If None (default) a new
        figure will be created.

    Returns
    -------
    The matplotlib axes objects plotted into.
    (optional: the design_conditions if not None, yet to implement)

    """

    # design conditions can be True or array
    n_dim = 2
    if swap_axis:
        x_idx = 1
        y_idx = 0
    else:
        x_idx = 0
        y_idx = 1

    if semantics is None:
        semantics = get_default_semantics(n_dim)

    if ax is None:
        _, ax = plt.subplots()

    if design_conditions:
        try:  # if iterable assume it's already the design conditions
            iter(design_conditions)
        except TypeError:  # if it is not an array we compute the default design_conditions
            design_conditions = calculate_design_conditions(
                contour, swap_axis=swap_axis
            )

        ax.scatter(
            design_conditions[:, 0],
            design_conditions[:, 1],
            c="#DDAA33",
            marker="x",
            zorder=2.5,
        )

    if sample is not None:
        sample = np.asarray(sample)
        ax.scatter(
            sample[:, x_idx],
            sample[:, y_idx],
            c="k",
            marker=".",
            alpha=0.3,
            rasterized=True,
        )

    coords = contour.coordinates
    x = coords[:, x_idx].tolist()
    x.append(x[0])
    y = coords[:, y_idx].tolist()
    y.append(y[0])

    # It was thought that this line caused a DepreciationWarning, but the change
    # was reverted as we were not sure about the reason.
    # https://github.com/virocon-organization/virocon/commit/45482e0b5ff2d21c594f0e292b3db9c971881b5c
    # https://github.com/virocon-organization/virocon/pull/124#discussion_r684193507
    ax.plot(x, y, c="#BB5566")
    # ax.plot(np.asarray(x, dtype=object), np.asarray(y, dtype=object), c="#BB5566")

    x_name = semantics["names"][x_idx]
    x_symbol = semantics["symbols"][x_idx]
    x_unit = semantics["units"][x_idx]
    x_label = f"{x_name}," + r" $\it{" + f"{x_symbol}" + r"}$" + f" ({x_unit})"
    y_name = semantics["names"][y_idx]
    y_symbol = semantics["symbols"][y_idx]
    y_unit = semantics["units"][y_idx]
    y_label = f"{y_name}," + r" $\it{" + f"{y_symbol}" + r"}$" + f" ({y_unit})"
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if design_conditions is None:
        return ax
    else:
        return ax, design_conditions
