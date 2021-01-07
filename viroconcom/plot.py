#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots datasets, model fits and contour coordinates.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings

__all__ = ["plot_sample", "plot_marginal_fit", "plot_dependence_functions",
           "plot_contour", "SamplePlotData", "plot_confidence_interval",
           "plot_wave_breaking_limit", "hs_from_limiting_sig_wave_steepness"]


def plot_sample(sample_plot_data, ax=None, do_plot_rasterized=True):
    """
    Plots the sample of metocean data.

    Parameters
    ----------
    sample_plot_data : SamplePlotData,
        The sample that should be plotted and its meta information.
    """
    if ax is None:
        ax = sample_plot_data.ax
    ps = sample_plot_data
    x = ps.x
    y = ps.y
    if ps.x_inside is not None and ps.y_inside is not None:
        inside_label = 'inside contour'
        outside_label = 'outside contour'
        ax.scatter(ps.x_inside, ps.y_inside, s=11, alpha=0.5, c='k',
                      marker='o', label=inside_label, rasterized=do_plot_rasterized)
        ax.scatter(ps.x_outside, ps.y_outside, s=9, alpha=0.5, c='r',
                      marker='D', label=outside_label, rasterized=do_plot_rasterized)
    else:
        if ps.label:
            ax.scatter(x, y, s=40, alpha=0.5, c='k', marker='.',
                          label=ps.label, rasterized=do_plot_rasterized)
        else:
            ax.scatter(x, y, s=40, alpha=0.5, c='k', marker='.',
                       label='observation', rasterized=do_plot_rasterized)

    # Remove axis on the right and on the top (Matlab 'box off').
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def plot_marginal_fit(sample, dist, fig, ax=None, label=None, color_sample='k',
                      marker_sample='x', marker_size_sample=3, color_fit='b',
                      dataset_char=None, legend_fontsize=8):
    """
    Plots the fitted marginal distribution versus a dataset in a quantile-
    quantile (QQ) plot.

    Parameters
    ----------
    sample : ndarray of floats
        The environmental data sample that should be plotted against the fit.
    dist : Distribution
        The distribution that has been fitted.
    fig : matplotlib Figure
        Figure object that shall be used for the plot.
    ax : matplotlib Axes
        Axes object on the figure that shall be used for the plot.
    label : string
        Description of the random variable / sample, e.g. '$h_s$ (m)'.
    color_sample : color (char, string or RGB)
        Character that represents in a color using the matplotlib conventions.
    marker_sample : char
        Character that represents a marker using the matplotlib conventions.
    marker_size_sample : int
        Number that specifies the marker's size using the matplotlib conventions.
    color_fit :  color (char, string or RGB)
        Character that represents in a color using the matplotlib conventions.
    dataset_char : char
        Character, which is the name of the dataset, e.g. 'A'
    legend_fontsize int,
        Fontsize of the legend text.
    """
    if ax is None:
        ax = fig.add_subplot(111)
    plt.sca(ax)
    stats.probplot(sample, dist=dist, plot=ax)
    ax.get_lines()[0].set_markerfacecolor(color_sample)
    ax.get_lines()[0].set_markeredgecolor(color_sample)
    ax.get_lines()[0].set_marker(marker_sample)
    ax.get_lines()[0].set_markersize(marker_size_sample)
    ax.get_lines()[1].set_color(color_fit)
    ax.title.set_text('')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #if max(sample) < 12:
        #plt.xlim((0, 12))
        #plt.ylim((0, 15.5))
    #else:
        #plt.xlim((0, 35))
        #plt.ylim((0, 35))
    if dist.name == 'ExponentiatedWeibull':
        dist_description = 'Exponentiated Weibull\n' \
                           '($\\alpha$=' + str('%.3g' % dist.scale(0)) + ', ' \
                           '$\\beta$=' + str('%.3g' % dist.shape(0)) + ', ' \
                           '$\\delta$=' + str('%.3g' % dist.shape2(0)) +')'
    elif dist.name == 'Weibull':
        dist_description = 'Weibull, ' \
                           '$\\alpha$=' + str('%.3g' % dist.scale(0)) + ', ' \
                           '$\\beta$=' + str('%.3g' % dist.shape(0))
    else:
        dist_description = dist.name
    plt.legend(['Dataset '+ dataset_char, dist_description], loc='upper left',
               frameon=False, prop={'size': legend_fontsize})
    xlabel_string = 'Theoretical quantiles'
    ylabel_string = 'Ordered values'
    if label:
        xlabel_string += ', ' + str(label).lower()
        ylabel_string += ', ' + str(label).lower()
    plt.xlabel(xlabel_string)
    plt.ylabel(ylabel_string)


def plot_dependence_functions(
        fit, fig, ax1=None, ax2=None, unconditonal_variable_label=None,
        marker_discrete='o', markersize_discrete=5,
        markerfacecolor_discrete='lightgray', markeredgecolor_discrete='k',
        style_dependence_function='b-', legend_fontsize=8,
        factor_draw_longer=1.1):
    """
    Plots the fitted dependence function using two subplots, one subplot showing
    the fit of the shape value and one subplot showing the fit of the scale
    value.

    This funciton only works if the conditional distribution is a Weibull
    distribution or a lognormal disribution.

    Parameters
    ----------
    fit : Fit
    fig : matplotlib Figure
        Figure object that shall be used for the plot.
    ax1 : matplotlib Axes, defeaults to None
        Axes object on the figure that shall be used for the plot.
    ax2 : matplotlib Axes, defeaults to None
        Axes object on the figure that shall be used for the plot.
    unconditonal_variable_label : str, defaults to None
    marker_discrete : char, defaults to 'o'
    markersize_discrete : int, defaults to 5
    markerfacecolor_discrete : color (char, string or RGB), defaults to 'lightgray'
    markeredgecolor_discrete : color (char, string or RGB), defaults to 'k'
    style_dependence_function : str, defaults to 'b-'
        Style of the fitted dependence function.
    legend_fontsize : int, defaults to 8
        Font size of the legend's text.
    factor_draw_longer : float, defaults to 1.1
        How much longer than the last point should the fitted curve be plotted.

    Raises
    ------
    NotImplementedError
        If the distribution that shall be plotted is not supported yet.
    """

    supported_dists = ['ExponentiatedWeibull', 'Lognormal', 'Weibull']
    if fit.mul_var_dist.distributions[1].name not in supported_dists:
        raise NotImplementedError(
            'The distribution you tried to plot is not not supported in '
            'plot_dependence_functions. You used the distribution {}'
            ' .'.format(fit.mul_var_dist.distributions[1]))


    if ax1 is None:
        ax1 = fig.add_subplot(121)
    if ax2 is None:
        ax2 = fig.add_subplot(122)

    plt.sca(ax1)
    scale_at = fit.multiple_fit_inspection_data[1].scale_at
    x1 = np.linspace(0, max(scale_at) * factor_draw_longer, 100)
    if fit.mul_var_dist.distributions[1].scale.func_name == 'power3':
        dp_function = r'$' + str('%.3g' % fit.mul_var_dist.distributions[1].scale.a) + \
                      r'+' + str('%.3g' % fit.mul_var_dist.distributions[1].scale.b) + \
                      r'\cdot h_s^{' + str('%.3g' % fit.mul_var_dist.distributions[1].scale.c) + '}$'
    elif fit.mul_var_dist.distributions[1].scale.func_name == 'lnsquare2':
        dp_function = r'$\ln(' + str('%.3g' % fit.mul_var_dist.distributions[1].scale.a) + \
                      r'+' + str('%.3g' % fit.mul_var_dist.distributions[1].scale.b) + \
                      r'\sqrt{h_s / g})$'
    elif fit.mul_var_dist.distributions[1].scale.func_name == 'alpha3':
        dp_function = r'$(' + str('%.3g' % fit.mul_var_dist.distributions[1].scale.a) + \
                      r'+' + str('%.3g' % fit.mul_var_dist.distributions[1].scale.b) + \
                      r'\cdot v^{' + str('%.3g' % fit.mul_var_dist.distributions[1].scale.c) + \
                      r'}) / 2.0445^{(1 / \beta_{hs})}$'
    else:
        dp_function = str(fit.mul_var_dist.distributions[1].scale)

    if fit.mul_var_dist.distributions[1].name == 'Lognormal':
        plt.plot(scale_at, np.log(fit.multiple_fit_inspection_data[1].scale_value),
                 marker_discrete,
                 markersize=markersize_discrete,
                 markerfacecolor=markerfacecolor_discrete,
                 markeredgecolor=markeredgecolor_discrete,
                 label='from marginal distribution')
        plt.plot(x1, np.log(fit.mul_var_dist.distributions[1].scale(x1)),
                 style_dependence_function, label=dp_function)
        ylabel = '$μ$'
    if fit.mul_var_dist.distributions[1].name == 'Weibull' or \
                    fit.mul_var_dist.distributions[1].name == 'ExponentiatedWeibull':
        plt.plot(scale_at, fit.multiple_fit_inspection_data[1].scale_value,
                 marker_discrete,
                 markersize=markersize_discrete,
                 markerfacecolor=markerfacecolor_discrete,
                 markeredgecolor=markeredgecolor_discrete,
                 label='from marginal distribution')
        plt.plot(x1, fit.mul_var_dist.distributions[1].scale(x1),
                 style_dependence_function, label=dp_function)
        ylabel = '$α$'
    plt.xlabel(unconditonal_variable_label)
    plt.legend(frameon=False, prop={'size': legend_fontsize})
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.ylabel(ylabel)

    plt.sca(ax2)
    shape_at = fit.multiple_fit_inspection_data[1].shape_at
    x1 = np.linspace(0, max(shape_at) * factor_draw_longer, 100)
    plt.plot(shape_at, fit.multiple_fit_inspection_data[1].shape_value,
             marker_discrete,
             markersize=markersize_discrete,
             markerfacecolor=markerfacecolor_discrete,
             markeredgecolor=markeredgecolor_discrete,)
    plt.plot(x1, fit.mul_var_dist.distributions[1].shape(x1),
             style_dependence_function)
    plt.xlabel(unconditonal_variable_label)
    if fit.mul_var_dist.distributions[1].name == 'Lognormal':
        ylabel = '$σ$'
        if fit.mul_var_dist.distributions[1].shape.func_name == 'exp3':
            dp_function = r'$' + str('%.3g' % fit.mul_var_dist.distributions[1].shape.a) + \
                          r'+' + str('%.3g' % fit.mul_var_dist.distributions[1].shape.b) + \
                          r'\exp (' + str('%.3g' % fit.mul_var_dist.distributions[1].shape.c) + \
                          r'h_s)$'
        elif fit.mul_var_dist.distributions[1].shape.func_name == 'powerdecrease3':
            dp_function = r'$' + str('%.4f' % fit.mul_var_dist.distributions[1].shape.a) + \
                          r'+ 1 / (h_s + ' + str('%.3g' % fit.mul_var_dist.distributions[1].shape.b) + \
                          r')^{' + str('%.3g' % fit.mul_var_dist.distributions[1].shape.c) + \
                          r'}$'
        elif fit.mul_var_dist.distributions[1].shape.func_name == 'asymdecrease3':
            dp_function = r'$' + str('%.4f' % fit.mul_var_dist.distributions[1].shape.a) + \
                          r' + ' + str('%.3g' % fit.mul_var_dist.distributions[1].shape.b) + \
                          r' / (1 + ' + str('%.3g' % fit.mul_var_dist.distributions[1].shape.c) + \
                          r' h_s )$'
    if fit.mul_var_dist.distributions[1].name == 'Weibull'  or \
                    fit.mul_var_dist.distributions[1].name == 'ExponentiatedWeibull':
        ylabel = '$β$'
        if fit.mul_var_dist.distributions[1].shape.func_name == 'power3':
            dp_function = r'$' + str('%.4f' % fit.mul_var_dist.distributions[1].shape.a) + \
                          r'+' + str('%.3g' % fit.mul_var_dist.distributions[1].shape.b) + \
                          r'\cdot h_s^{' + str('%.3g' % fit.mul_var_dist.distributions[1].shape.c) + '}$'
        elif fit.mul_var_dist.distributions[1].shape.func_name == 'logistics4':
            # logistics4 uses np.abs(c), to display it nicer, abs(c) is shown.
            absOfC = np.abs(fit.mul_var_dist.distributions[1].shape.c)
            dp_function = r'$' + str('%.3g' % fit.mul_var_dist.distributions[1].shape.a) + \
                          r'+' + str('%.3g' % fit.mul_var_dist.distributions[1].shape.b) + \
                          r'/ [1 + e^{-' + str('%.3g' % absOfC) + \
                          r'(v - ' + str('%.3g' % fit.mul_var_dist.distributions[1].shape.d) + \
                          r')}]$'
        else:
            dp_function = str(fit.mul_var_dist.distributions[1].shape)
    plt.legend(['from marginal distribution', dp_function], frameon=False, prop={'size': legend_fontsize})
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.ylabel(ylabel)


def plot_contour(x, y, ax=None, contour_label=None, x_label=None, y_label=None,
                 style='b-', color=None, linewidth=None, linestyle=None, alpha=1,
                 sample_plot_data=None, x_lim=None, upper_ylim=None,
                 median_x=None, median_y=None, median_style='r-',
                 median_label='median of x2|x1', line_style=None):
    """
    Plots the environmental contour.

    The method expects the coordinates to be ordered by angle.

    Parameters
    ----------
    x : ndarray of floats
        The contour's coordinates in the x-direction.
    y : ndarray of floats
        The contour's coordiantes in the y-direction.
    ax : Axes, optional (defaults to None)
        Axes of the figure where the contour should be plotted.
        If None a figure is created here.
    contour_label : str, optional (defaults to None)
        The environmental contour's label that will be used in the legend.
    x_label : str, optional (defaults to None)
        Label for the x-axis.
    y_label : str, optional (defaults to None)
        Label for the y-axis.
    style : str, optional (defaults to 'b-')
        Matplotlib style abbrevation. Will be ignored if 'color', 'lindwidth'
        or 'linestyle' is pecified.
    color : any matplotlib color, optional (defaults to None)
        Color of the line.
    linewidth : float value in points, optional (defaults to None)
        Width of the line.
    linestyle : any matplotlib linestyle, optional (defaults to None)
        Style of the line.
    alpha : float, optional (default to 1)
        Alpha value (transparency) for the contour's line.
    sample_plot_data : SamplePlotData, optional (defaults to None)
        The sample that should be plotted and its meta information.
    x_lim : tuple of floats, optional (defaults to None)
        x-Axis limit.
    upper_ylim : float, optional (defaults to None)
        y-Axis limit.
    median_x : ndarray of floats, optional (defaults to None)
        If the median of x2|x1 should be plotted, these are the x-values.
    median_y : ndarray of floats, optional (defaults to None)
        If the median of x2|x1 should be plotted, these are the y-values.
    median_style : str, optional (defaults to 'r-')
        Matplotlib line style for plotting the median of x2|x1.
    median_label : str, optional (defaults to 'median of x2|x1')
        Label for the legend of the plotted median line.
    """
    if line_style:
        warnings.warn("Keyword 'line_style' is depreciated and will be removed "
                      "in future versions. Use keyword 'style' instead.", DeprecationWarning)
        style = line_style

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # For generating a closed contour: add the first coordinate at the end.
    xplot = x.tolist()
    xplot.append(x[0])
    yplot = y.tolist()
    yplot.append(y[0])

    # Plot the contour and, if provided, also the sample.
    if sample_plot_data:
        plot_sample(sample_plot_data, ax=ax)
    if color is None and linewidth is None and linestyle is None:
        ax.plot(xplot, yplot, style, alpha=alpha, label=contour_label)
    else:
        ax.plot(xplot, yplot, color=color, linewidth=linewidth,
                linestyle=linestyle, alpha=alpha,
                label=contour_label)
    if median_x is not None:
        ax.plot(median_x, median_y, median_style, label=median_label)

    # Format the figure.
    if contour_label:
        ax.legend(loc='upper left', frameon=False)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if x_lim:
        ax.set_xlim(x_lim)
    y_lim_factor = 1.2
    if sample_plot_data and upper_ylim is None:
        # If there is not enough space for the legend in the upper left corner:
        # make space for it.
        max_index = np.where(sample_plot_data.y == max(sample_plot_data.y))
        if sample_plot_data.x[max_index] < 0.6 * max(max(x), max(sample_plot_data.x)):
            y_lim_factor = 1.35

        upper_ylim = max(max(y), max(sample_plot_data.y)) * y_lim_factor
    elif upper_ylim is None:
        upper_ylim = max(y) * y_lim_factor
    ax.set_ylim((0, upper_ylim))


    # Remove axis on the right and on the top (Matlab 'box off').
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


class SamplePlotData():
    """
    Class that holds a plotted sample and its meta information.

    Attributes
    ----------
    x : ndarray of floats
        The sample's first environmental variable.
    y : ndarray of floats
        The sample's second environmental variable.
    ax : Axes
        Axes of the figure where the scatter plot should be drawn.
    label : str
        Label that will be used in the legend for the sample.
    x_inside : ndarray of floats
        Values in the first dimension of the points inside the contour.
    y_inside : ndarray of floats
        Values in the second dimension of the points inside the contour.
    x_outside : ndarray of floats
        Values in the first dimension of the points outside the contour.
    y_outside : ndarray of floats
        Values in the second dimension of the points outside the contour.
    return_period : int, optional
        Return period in years. Is used in legend for describing the inside and
        outside datapoints.
    """

    def __init__(self, x, y, ax=None, label=None, x_inside=None, y_inside=None,
                 x_outside=None, y_outside=None, return_period=None):
        """
        Parameters
        ----------
        x : ndarray of floats
            The sample's first environmental variable.
        y : ndarray of floats
            The sample's second environmental variable.
        ax : Axes, optional (defaults to None)
            Axes of the figure where the scatter plot should be drawn.
        label : str, optional (defaults to None)
            Label that will be used in the legend for the sample.
        x_inside : ndarray of floats, optional (defaults to None)
            Values in the first dimension of the points inside the contour.
        y_inside : ndarray of floats, optional (defaults to None)
            Values in the second dimension of the points inside the contour.
        x_outside : ndarray of floats, optional (defaults to None)
            Values in the first dimension of the points outside the contour.
        y_outside : ndarray of floats, optional (defaults to None)
            Values in the second dimension of the points outside the contour.
        return_period : int, optional (defaults to None)
            Return period in years. Is used in legend for describing the inside and
            outside datapoints.
        """
        self.x = x
        self.y = y
        self.ax = ax
        self.label = label
        self.x_inside = x_inside
        self.y_inside = y_inside
        self.x_outside = x_outside
        self.y_outside = y_outside
        self.return_period = return_period


def plot_confidence_interval(x_median, y_median, x_bottom, y_bottom, x_upper,
                             y_upper, ax, x_label=None, y_label=None,
                             contour_labels=[None, None, None], sample_plot_data=None):
    """
    Plots the confidence interval (median, bottom, upper) in a standardized
    appearance.
    Parameters
    ----------
    x_median : ndarray of floats
        The 50-percentile contour's coordinates in the x-direction.
    y_median : ndarray of floats
        The 50-percentile contour's coordinates in the y-direction.
    x_bottom : ndarray of floats
        The bottom percentile contour's coordinates in the x-direction.
    y_bottom : ndarray of floats
        The bottom percentile contour's coordinates in the y-direction.
    x_upper : ndarray of floats
        The upper percentile contour's coordinates in the x-direction.
    y_upper : ndarray of floats
        The upper percentile contour's coordinates in the y-direction.
    ax : Axes
        Axes of the figure where the contour should be plotted.
    x_label : str, optional (defaults to None)
        Label for the x-axis.
    y_label : str, optional (defaults to None)
        Label for the y-axis.
    contour_labels : list of str, optional (defaults to [None, None, None])
        Label for the environmental contours that will be used in the legend.
    sample_plot_data : SamplePlotData, optional
        If provided, this sample is plotted together with the contours.
    """
    plot_contour(x=x_median,
                 y=y_median,
                 ax=ax,
                 x_label=x_label,
                 y_label=y_label,
                 style='b-',
                 contour_label=contour_labels[0],
                 sample_plot_data=sample_plot_data)
    plot_contour(x=x_bottom,
                 y=y_bottom,
                 contour_label=contour_labels[1],
                 style='r--',
                 ax=ax)
    plot_contour(x=x_upper,
                 y=y_upper,
                 contour_label=contour_labels[2],
                 style='r--',
                 ax=ax)


def plot_wave_breaking_limit(ax, bottom_tz=0, upper_tz=20, steps=100):
    """
    Plots the wave breaking limit on a given axes.

    Assumes that x = zero-up-crossing period and y = sig. wave height.

    Parameters
    ----------
    ax : Axes
    bottom_tz : float (defaults to 0)
        Bottom limit for the curve.
    upper_tz : float (defaults to 20)
        Upper limit for the curve.
    steps : int (defaults to 100)
        Number of points that are plotted on the curve.
    """
    tz_lim = np.linspace(bottom_tz, upper_tz, steps)
    hs_lim = hs_from_limiting_sig_wave_steepness(tz_lim)
    ax.plot(tz_lim, hs_lim, linestyle='-.', color=[0.5, 0.5, 0.5])


def hs_from_limiting_sig_wave_steepness(tz):
    """
    Calculates highest Hs value for a given Tz based on wave steepness.

    The calculaion uses the 'limiting significant wave steepness' described in
    DNV GL's DNV-GL-RP-C205:2017 (section 3.5.3.5. and 3.5.4)

    Parameters
    ----------
    tz : ndarray of floats
        Zero-up-crossing period in seconds.
    Returns
    -------
    hs : ndarray of floats
        Significant wave height in meters.
    """
    TZLOW_STEEPNESS_VALUE = 0.1 # From DNVG-GL-RP-C205:2017, 3.5.4 .
    TZHIGH_STEEPNESS_VALUE = 0.0666666  # From DNVG-GL-RP-C205:2017, 3.5.4 .

    G = 9.81

    ss = np.empty(tz.size)
    ss[:] = np.nan


    for i in range(tz.size):
        if tz[i] <= 6:
            ss[i] = 0.1
        elif tz[i] < 12:
            ss[i] = \
                TZLOW_STEEPNESS_VALUE\
                + (TZHIGH_STEEPNESS_VALUE - TZLOW_STEEPNESS_VALUE) \
                  / 6.0 * (tz[i] - 6)
        else:
            ss[i] = TZHIGH_STEEPNESS_VALUE

    hs = ss * tz**2 * G / (2 * np.pi)

    return hs
