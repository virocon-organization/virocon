***********************
Description Cheat Sheet
***********************

dist_descriptions
~~~~~~~~~~~~~~~~~

dist_descriptions are used to create joint models.
A list of dist_descriptions is passed to
:class:`~virocon.jointmodels.GlobalHierarchicalModel`, one for each dimension.

For an independent distribution:

.. code:: python

    # general
    dist_description = {
        "distribution": <distribution instance>,
        "intervals": <interval slicer instance>,
    }

    # example
    dist_description_v = {
        "distribution": ExponentiatedWeibullDistribution(),
        "intervals": WidthOfIntervalSlicer(2, min_n_points=50),
    }

For a conditional distribution:

.. code:: python

    # general
    dist_description = {
        "distribution": <distribution instance>,
        "conditional_on": <index of distribution this dist depends on>,
        "parameters": <dict with parameter names as keys
                       and dependence function instances as values>,
    }

    # example
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

    # general
    fit_description = {
        "method": <method keyword>,
        "weights": <define the weights, if "wlsq" is used>
    }

    # example
    fit_description_hs = {"method": "wlsq", "weights": "quadratic"}



semantics
~~~~~~~~~

The semantics dict describes the semantics of the model.
It's content is not interpreted, but rather passed to e.g. plotting functions,
where it is used to set label texts.
Each key contains a list.
Each list contains one entry for each dimension of the model.

.. code:: python

    semantics = {
        "names": <list of names of the modeled variables>,
        "symbols": <list of symbols used for the modeled variables>,
        "units": <list of unit strings of the modeled variables>,
    }

    semantics = {
        "names": ["Mean wind speed", "Significant wave height"],
        "symbols": ["V", "H_s"],
        "units": ["m s$^{-1}$", "m",],
    }
