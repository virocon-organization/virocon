.. _definitions:

*****************************************
Environmental contour: Useful definitions
*****************************************

A contour implements a method to define multivariate extremes based on a joint probabilistic model of variables like
significant wave height, wind speed or spectral peak period. Contour curves or surfaces for more than two environmental
parameters give combination of environmental parameters which approximately describe the various actions corresponding
to the given exceedance probability [1]_.

**Exceedance probability**

Probability that an environmental contour is exceeded (exceedance probability). The exceedance probability, α,
corresponds to a certain recurrence or return period, Tr, which describes the average time period between two
consecutive environmental states that exceed the contour. In the following formula, the return period Tr is given in years, while Ts, the
state duration of independent events is given in hours.

:math:`\alpha= \frac{T_s}{T_r * 365.25 * 24}`

Note that exceedance can be defined in various ways for environmental contours
(Mackay and Haselsteiner, 2021) [2]_ .

**State duration Ts**

Time period for which an environmental state is measured, usually expressed in hours.

**Return period Tr**

Describes the average time period between two consecutive environmental states that exceed a contour.
In the univariate case the contour is a threshold.

**Different types of environmental contours**

The figure below visualizes different types of environmental contours [2]_.

.. figure:: contour_types.png
    :scale: 50 %
    :alt: Different types of environmental contours [2]_.



.. [1] NORSOK standard N-003, Edition 2, September 2007, Actions and action effects.
.. [2] Mackay, E., & Haselsteiner, A. F. (2021): Marginal and total exceedance probabilities of environmental contours. Marine Structures, 75. https://doi.org/10.1016/j.marstruc.2020.102863