*********************************
Fit your own data with fitting.py
*********************************

This module implements the possibility to fit a probabilistc model to your own data.

If you want to fit multivariate distribution to your data set you have to build an object of the class ``Fit`` in this module.
The necessary parameters for building objects of ``Fit`` are also listed in the documentation of this class.
Exemplary call::

    example_fit = Fit((data_1, data_2), (dist_description_0, dist_description_1), 15)

It's important that the parameter ``samples`` is in the form (sample_1, sample_2, ...).
Each sample is a collection of data from type *list* and also all samples have the same length. The parameter ``dist_descriptions``
should be from type *list* and contains a dictionary for each dimension in the same sequence of the samples. It should accordingly have
the same length as ``samples``. The last parameter ``n_steps`` indicates how many distributions should be fitted for a dependent parameter.
For each distribution an additional parameter is created which affects the final fit.

.. image:: fit_scale.png

Each ``dist_description`` contains the name of the current distribution (i.e. ``"Weibull"``). Then it contains the dependency for this dimension
from type *list*. In the sequence of ``shape, loc, scale`` it contains integers for the dependency of the current parameter or *None* if it has no
dependency. It is important that the dependency is less than the index of the current dimension. The list for the parameter ``functions`` also has length of three
and contains information about the used functions for fitting. Actually you can switch between the functions:

- **f1** :  :math:`a + b * x^c`
- **f2** : :math:`a + b * e^{x * c}`
- **None** : no dependency

Example for ``dist_description``::

	example_dist_description = {'name': 'Lognormal_1', 'dependency': (0, None, 1),
				                'functions': ('f1', None, 'f2')}

If the fit is finished it has the attribute ``mul_var_dist`` that is an object of ``MultivariateDistribution`` that contains all distributions you
can use to build a contour for your data. Also it has the attributes ``mul_param_points`` and ``mul_dist_points`` which can be used to visualize
your fit.

.. todo::
    adapt to changes in fitting.py
