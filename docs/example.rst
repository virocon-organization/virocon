**************************
Example: Sea state contour
**************************

Let's start this user guide with a simple example.

Based on a dataset, the long-term joint distribution of sea states is estimated
and this distribution will be used to construct an environmental contour with a
return period of 50 years. This example follows the OMAE2020 model by
Haselsteiner et al. (2020) ::

    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt

    from virocon import (GlobalHierarchicalModel, get_OMAE2020_Hs_Tz,
                         calculate_alpha, IFORMContour, plot_2D_contour)

    # Load sea state measurements from the NDBC buoy 46025.

    data = pd.read_csv("datasets/NDBC_buoy_46025.csv", sep=",")[["Hs", "T"]]

    # Define the structure of the probabilistic model that will be fitted to the
    # dataset. This model structure has been proposed in the paper "Global
    # hierarchical models for wind and wave contours: Physical interpretations
    # of the dependence functions" by Haselsteiner et al. (2020).

    dist_descriptions, fit_descriptions, model_description = get_OMAE2020_Hs_Tz()

    # Fit the model to the data.

    ghm = GlobalHierarchicalModel(dist_descriptions)
    ghm.fit(data, fit_descriptions=fit_descriptions)

    # Compute an IFORM contour with a return period of 50 years.

    state_duration = 1 # Sea state duration in hours
    return_period = 50 # Return period in years
    alpha = calculate_alpha(state_duration, return_period)
    my_iform = IFORMContour(ghm, alpha)
    my_coordinates = my_iform.coordinates

    # Plot the data and the contour.


    ax = plot_2D_contour(my_iform, model_desc=model_description, sample=data,
                     design_conditions=True, swap_axis=True)

The code, which is available as a Python file here_, will create this plot:

.. figure:: sea_state_contour.png
    :scale: 100 %
    :alt: sea state contour

    Environmental contour with a return period of 50 years.

.. _here: https://github.com/virocon-organization/viroconcom/blob/master/examples/sea_state_iform_contour.py
