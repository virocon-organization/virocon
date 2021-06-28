**************************************
Quick start example: Sea state contour
**************************************

Let’s start this user guide with a simple example, which can be copied easily line by line or downloaded from the
examples section of the repository. Based on a dataset, the long-term joint distribution of sea states is estimated,
and this distribution will be used to construct an environmental contour with a return period of 50 years. This example
follows the OMAE2020 model by Haselsteiner et al. (2020) [1]_. The following steps are carried out in the following quick
start example:

1.	Load the environmental data that should be described by the joint model.
2.	Use a predefined model to describe the dependence structure and define the parametric distributions.
3.	Estimate the parameter values of the MultivariateModel (fitting).
4.	Define the contour’s return period and environmental state duration.
5.	Choose a type of contour: IFormContour, ISormContour, DirectSamplingContour or HighestDensityContour.


Import the required virocon packages.
::

    from virocon import (GlobalHierarchicalModel, get_OMAE2020_Hs_Tz,
                         calculate_alpha, IFORMContour, plot_2D_contour)

Import packages that are needed additionally.
::

    from matplotlib import pyplot as plt

Load the sea state data set. Here, we use a dataset used in a benchmark study which was published at the OMAE 2019
conference by Haselsteiner et. Al. (2019) [2]_.
::

    data = pd.read_csv("datasets/NDBC_buoy_46025.csv", sep=",")[["Hs", "T"]]

Use a predefined model to describe the dependence structure and define the structure of the joint distribution model.
virocon provides 4 simple predefined models, which can be directly executed. Here, we use the Hs-Tz joint distribution
model recommended in DNVGL-RP-C203 (2017) [3]_.
::

    dist_descriptions, fit_descriptions, semantics = get_OMAE2020_Hs_Tz()

Estimate the values of the model’s parameter. This step is also well known as “fitting”.
::

    ghm = GlobalHierarchicalModel(dist_descriptions)
    ghm.fit(data, fit_descriptions=fit_descriptions)

Compute the contour: Define the contour’s return period and environmental state duration. Choose a type of contour:
IFormContour, ISormContour, DirectSamplingContour or HighestDensityContour. Plot the contour.
::
    state_duration = 1 # Sea state duration in hours
    return_period = 50 # Return period in years
    alpha = calculate_alpha(state_duration, return_period)
    my_iform = IFORMContour(ghm, alpha)
    my_coordinates = my_iform.coordinates

    # Plot the data and the contour.


    ax = plot_2D_contour(my_iform, semantics=semantics, sample=data,
                         swap_axis=True)

The code, which is available as a Python file here_, will create this plot:

.. figure:: sea_state_contour.png
    :scale: 100 %
    :alt: sea state contour

    Environmental contour with a return period of 50 years.

.. _here: https://github.com/virocon-organization/virocon/blob/master/examples/hstz_contour_simple.py
.. [1] •	Haselsteiner et. Al. (2020): Haselsteiner, A.F.; Sander, A.; Ohlendorf, J.H.; Thoben, K.D. (2020). Global hierarchical models for wind and wave contours: physical interpretations of the dependence functions. OMAE 2020, Fort Lauderdale, USA. Proceedings of the 39th International Conference on Ocean, Offshore and Arctic Engineering.
.. [2] •	Haselsteiner et. Al. (2019): Haselsteiner, A.F.; Coe, R.; Manuel, L.; Nguyen, P.T.T.; Martin, N.; Eckert-Gallup, A. A benchmarking exercise on estimating extreme environmental conditions: methodology and baseline results. Proceedings of the 38th International Conference on Ocean, Offshore and Arctic Engineering OMAE2019, June 09-14, 2019, Glasgow, Scotland.
.. [3] •	DNV GL. (2017). Recommended practice DNVGL-RP-C205: Environmental conditions and environmental loads.