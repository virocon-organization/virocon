*****************
Detailed examples
*****************

This chapter will explain how the structure of the joint distribution model is created in virocon. The process of
estimating the parameter values of a joint distribution, the “fitting” is explained in more detail by means of two
examples. To create an environmental contour, first, we need to define a joint distribution. Then, we can choose a
specific contour method and initiate the calculation. virocon uses so-called global hierarchical models to define the
joint distribution and offers four common methods how an environmental contour can be defined based on a given joint
distribution. Generally, virocon provides two ways of creating a joint model and calculating a contour. The first option
is using an already predefined model, which was explained before in the quick start section. The second option is
defining a custom statistical model.

If the joint distribution is known, the procedure of calculating an environmental contour with virocon can be summarized as:

1.	Load the environmental data that should be described by the joint model.
2.	Define the dependence structure of the joint model that we will use to describe the environmental data. To define a joint model, we define the univariate parametric distributions and the dependence structure. The dependence structure is defined using parametric functions.
3.	Define the univariate parametric distributions.
    a.	Create a first, independent univariate distribution.
    b.	Create another, usually dependent univariate distribution and define its dependency on the previous distributions.
    c.	Repeat step 2, until you have created a univariate distribution for each environmental variable.
4.	Estimate the parameter values of the MultivariateModel (fitting).
5.	Define the contour’s return period and environmental state duration.
6.	Choose a type of contour: IFormContour, ISormContour, DirectSamplingContour or HighestDensityContour.

50 year environmental contour with the Hs-Tz DNVGL model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, we use a sea state dataset with the variables Hs and Tz, fit the join distribution recommended in
DNVGL-RP-C203 [1]_ to it and compute an IFORM contour. This example reproduces the results published in
Haselsteiner et al. (2019) [2]_. Such a work flow is for example typical in ship design. The presented example can be
downloaded from the examples_ section of the repository.

**Imports**
::
    import numpy as np
    import matplotlib.pyplot as plt

    from virocon import (
    read_ec_benchmark_dataset,
    GlobalHierarchicalModel,
    WeibullDistribution,
    LogNormalDistribution,
    DependenceFunction,
    WidthOfIntervalSlicer,
    IFORMContour,
    plot_marginal_quantiles,
    plot_dependence_functions,
    plot_2D_contour,
    )

**Environmental data**

This dataset has been used in a benchmarking exercise, see https://github.com/ec-benchmark-organizers/ec-benchmark .
The dataset was derived from NDBC buoy 44007, see https://www.ndbc.noaa.gov/station_page.php?station=44007 .
::
    data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_A.txt")

**Dependence structure**

Define the structure of the joint model that we will use to describe the environmental data. To define a joint model, we
define the univariate parametric distributions and the dependence structure. The dependence structure is defined using
parametric functions.

A 3-parameter power function, which will be used as a dependence function.
::
    def _power3(x, a, b, c):
        return a + b * x ** c
A 3-parameter exponential function, which will be used a dependence function.
::
    def _exp3(x, a, b, c):
        return a + b * np.exp(c * x)


**Parametric distributions**
First, set lower and upper interval boundaries for the three parameter values.
::
    bounds = [(0, None), (0, None), (None, None)]

    power3 = DependenceFunction(_power3, bounds)
    exp3 = DependenceFunction(_exp3, bounds)

    dist_description_0 = {
        "distribution": WeibullDistribution(),
        "intervals": WidthOfIntervalSlicer(width=0.5),
    }
    dist_description_1 = {
        "distribution": LogNormalDistribution(),
        "conditional_on": 0,
        "parameters": {"mu": power3, "sigma": exp3}

**Multivariate Model**
::
    model = GlobalHierarchicalModel([dist_description_0, dist_description_1])

Semantics define a dictionary that describes the model. This description can be used while plotting the contour.
::
    semantics = {
    "names": ["Significant wave height", "Zero-crossing period"],
    "symbols": ["H_s", "T_z"],
    "units": ["m", "s"],
    }

Fit the model to the data (estimate the model's parameter values).
::
    model.fit(data)

Print the estimated parameter values.
::
    print(model)

Create plots to inspect the model's goodness-of-fit.
::
    fig1, axs = plt.subplots(1, 2, figsize=[10, 4.8])
    plot_marginal_quantiles(model, data, semantics, axes=axs)
    fig2, axs = plt.subplots(1, 2, figsize=[10, 4.8])
    plot_dependence_functions(model, semantics, axes=axs)

**Environmental contour**
Compute an IFORM contour with a return period of 20 years.
::
    state_duration = 1  # hours
    return_period = 20  # years
    alpha = state_duration / (return_period * 365.25 * 24)
    contour = IFORMContour(model, alpha)

Plot the contour on top of a scatter diagram of the metocean data.
::
    ax = plot_2D_contour(contour, sample=data, semantics=semantics, swap_axis=True)
    plt.show()



.. _examples: https://github.com/virocon-organization/virocon/tree/master/examples
.. [1] •	DNV GL. (2017). Recommended practice DNVGL-RP-C205: Environmental conditions and environmental loads.
.. [2] •	Haselsteiner et. Al. (2019): Haselsteiner, A.F.; Coe, R.; Manuel, L.; Nguyen, P.T.T.; Martin, N.; Eckert-Gallup, A. A benchmarking exercise on estimating extreme environmental conditions: methodology and baseline results. Proceedings of the 38th International Conference on Ocean, Offshore and Arctic Engineering OMAE2019, June 09-14, 2019, Glasgow, Scotland.


50 year V-Hs-Tz environmental contour
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::
    Stay tuned! We are currently working on this chapter.
    In the meantime if you have any questions feel free to open an issue.