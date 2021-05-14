ViroCon: viroconcom
===================

.. image:: https://travis-ci.org/virocon-organization/viroconcom.svg?branch=master
    :target: https://travis-ci.org/virocon-organization/viroconcom
    :alt: Build status

.. image:: https://coveralls.io/repos/github/virocon-organization/viroconcom/badge.svg?branch=master
    :target: https://coveralls.io/github/virocon-organization/viroconcom?branch=master

ViroCon is a software to compute environmental contours. `User Guide`_

About
-----

viroconcom is a Python software package to compute environmental contours. 

The software can support you to design marine structures, which need to withstand
load combinations based on wave, wind and current. It lets you define
extreme environmental conditions with a given return period using the
environmental contour method.

The following methods are implemented in viroconcom:

- Defining a joint probability distributions using a global hierarchical model structure
- Estimating the parameters of a global hierarchical model ("Fitting")
- Computing an environmental contour using either the

  - inverse first-order reliability method (IFORM),
  - inverse second-order reliability method (ISORM),
  - the direct sampling contour method or the
  - highest density contour method.

How to use viroconcom
---------------------
Requirements
~~~~~~~~~~~~
Make sure you have installed Python `3.8` by typing

.. code:: console

   python --version

in your `shell`_.

(Older version might work, but are not actively tested)

Consider using the python version management `pyenv`_.


Install
~~~~~~~
Install the latest version of viroconcom from PyPI by typing

.. code:: console

   pip install viroconcom


Alternatively, you can install from viroconcom repository’s Master branch
by typing

.. code:: console

   pip install https://github.com/virocon-organization/viroconcom/archive/master.zip


Usage
~~~~~

viroconcom is designed as an importable package.

The folder `examples`_ contains python files that show how one can
import and use viroconcom.

As an example, to run the file `sea_state_iform_contour.py`_, use
your shell to navigate to the folder that contains the file. Make sure
that you have installed matplotlib and run the Python file by typing

.. code:: console

   python sea_state_iform_contour.py

Our documentation contains a user guide, with  examples how to
`fit a distribution`_ to measurement data and how to
`compute environmental contours`_.

Documentation
-------------
**Learn.** Our `User Guide`_ covers installation, requirementss and overall work flow.

**Code.** The code’s documentation can be found `here`_.

**Paper.** Our `SoftwareX paper`_ "ViroCon: A software to compute multivariate
extremes using the environmental contour method." provides a concise
description of the software.

Contributing
------------

**Issue.** If you spotted a bug, have an idea for an improvement or a
new feature, please open a issue. Please open an issue in both cases: If
you want to work on in yourself and if you want to leave it to us to
work on it.

**Fork.** If you want to work on an issue yourself please fork the
repository, then develop the feature in your copy of the repository and
finally file a pull request to merge it into our repository.

**Conventions.** In our `Contribution Guide`_ we summarize our
conventions, which are consistent with PEP8.

Cite
----
If you are using viroconcom in your academic work please cite it by referencing
our SoftwareX paper.

Example: Environmental contours were computed using the package viroconcom
(version 1.2.0) of the software ViroCon [1].

[1] A.F. Haselsteiner, J. Lemkuhl, T. Pape, K.-L. Windmeier, K.-D. Thoben:
ViroCon: A software to compute multivariate extremes using the environmental
contour method. Accepted by SoftwareX.

License
-------

This software is licensed under the MIT license. For more information,
read the file `LICENSE`_.

.. _User Guide: https://virocon-organization.github.io/viroconcom/user_guide.html
.. _viroconweb: https://github.com/virocon-organization/viroconweb
.. _shell: https://en.wikipedia.org/wiki/Command-line_interface#Modern_usage_as_an_operating_system_shell
.. _pyenv: https://github.com/pyenv/pyenv
.. _www.python.org: https://www.python.org
.. _fit a distribution: https://virocon-organization.github.io/viroconcom/fitting.html
.. _compute environmental contours: https://virocon-organization.github.io/viroconcom/jointdistribution_and_contours.html
.. _examples: https://github.com/virocon-organization/viroconcom/tree/master/examples
.. _sea_state_iform_contour.py: https://github.com/virocon-organization/viroconcom/blob/master/examples/sea_state_iform_contour.py
.. _here: https://virocon-organization.github.io/viroconcom/
.. _Contribution Guide: https://virocon-organization.github.io/viroconcom/contributionguide.html
.. _LICENSE: https://github.com/virocon-organization/viroconcom/blob/master/LICENSE
.. _SoftwareX paper: https://github.com/ahaselsteiner/publications/blob/master/2018-10-25_SoftwareX_ViroCon_revised.pdf
