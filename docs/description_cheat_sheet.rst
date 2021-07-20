*************************************************
Cheat sheet for the "description" data structures
*************************************************

dist_descriptions
~~~~~~~~~~~~~~~~~

dist_descriptions are used to create joint models.

The dist_descriptions are dictionaries that include the description of the distributions. The dictionary has the
following keys depending on whether the distribution is conditional or not: "distribution", "intervals",
"conditional_on", "parameters". The key "intervals" is only used when describing unconditional distributions while the
keys "conditional_on" and "parameters" are only used when describing conditional variables. The key "distributions"
needs to be specified in both cases. With the key "distribution" an object of :class:`~virocon.distributions` is
committed. Here, we indicate the statistical distribution which describes the environmental variable best. In
"intervals" we indicate, which method should be used to split the range of values of the first environmental variable
into intervals. The conditional variable is then dependent on intervals of the first environmental variable. The key
"conditional_on" indicates the dependencies between the variables of the model. One entry per distribution/dimension.
It contains either None or int. If the first entry is None, the first distribution is unconditional. If the following
entry is an int, the following distribution depends on the first dimension as already described above. In "parameters"
we indicate the dependency functions that describe the dependency of the statistical parameters on the independent
environmental variable.

A list of dist_descriptions is passed to
:class:`~virocon.jointmodels.GlobalHierarchicalModel`, one for each dimension.

For an independent distribution:

.. code:: python

    # GENERAL
    dist_description = {
        "distribution": <distribution instance>,
        "intervals": <interval slicer instance>,
    }

    # EXAMPLE
    dist_description_v = {
        "distribution": ExponentiatedWeibullDistribution(),
        "intervals": WidthOfIntervalSlicer(2, min_n_points=50),
    }

For a conditional distribution:

.. code:: python

    # GENERAL
    dist_description = {
        "distribution": <distribution instance>,
        "conditional_on": <index of distribution this dist depends on>,
        "parameters": <dict with parameter names as keys
                       and dependence function instances as values>,
    }

    # EXAMPLE
    dist_description_hs = {
        "distribution": ExponentiatedWeibullDistribution(f_delta=5),
        "conditional_on": 0,
        "parameters": {"alpha": alpha_dep, "beta": beta_dep,},
    }


fit_descriptions
~~~~~~~~~~~~~~~~


fit_descriptions define which methods are used to estimate the parameter values
of a joint model, i.e. which fitting methods are used.
A list of fit_descriptions can be passed to the
:func:`virocon.jointmodels.GlobalHierarchicalModel.fit` method,
one for each dimension.
The list can also contain :code:`None`, in which case the defaults are used.
The supported method keywords depend on the distribution used.

.. code:: python

    # GENERAL
    fit_description = {
        "method": <method keyword>,
        "weights": <define the weights, if "wlsq" is used>
    }

    # EXAMPLE
    fit_description_hs = {"method": "wlsq", "weights": "quadratic"}



semantics
~~~~~~~~~

The semantics dict describes the semantics of the model. It's content is not interpreted, but rather passed to e.g.
plotting functions, where it is used to set label texts. Each key contains a list. Each list contains one entry for each
dimension of the model.

.. code:: python

    # GENERAL
    semantics = {
        "names": <list of names of the modeled variables>,
        "symbols": <list of symbols used for the modeled variables>,
        "units": <list of unit strings of the modeled variables>,
    }

    # EXAMPLE
    semantics = {
        "names": ["Mean wind speed", "Significant wave height"],
        "symbols": ["V", "H_s"],
        "units": ["m s$^{-1}$", "m",],
    }
