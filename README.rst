virocon
=======

.. image:: https://github.com/virocon-organization/virocon/actions/workflows/continuous_integration.yml/badge.svg
   :target: https://github.com/virocon-organization/virocon/actions/workflows/continuous_integration.yml
   :alt: GitHub Actions - CI
   
.. image:: https://img.shields.io/codecov/c/gh/virocon-organization/virocon
    :target: https://app.codecov.io/gh/virocon-organization/virocon
    :alt: Codecov
    
.. image:: https://readthedocs.org/projects/virocon/badge/?version=latest
   :target: https://virocon.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
    
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Black

virocon is a Python package to compute environmental contours. `User Guide`_

About
-----

virocon can support you to design marine structures, which need to withstand
load combinations based on wave, wind and current. It lets you define
extreme environmental conditions with a given return period using the
environmental contour method.

The following methods are implemented in virocon:

- Defining a joint probability distributions using a global hierarchical model structure
- Estimating the parameters of a global hierarchical model ("Fitting")
- Computing an environmental contour using either the

  - inverse first-order reliability method (IFORM),
  - inverse second-order reliability method (ISORM),
  - the direct sampling contour method or the
  - highest density contour method.
  - Futher, "AND" and "OR" exceedance contours can be calculated.

How to use virocon
------------------
Requirements
~~~~~~~~~~~~
Make sure you have installed Python `3.9` by typing

.. code:: console

   python --version

in your `shell`_.

(Older version might work, but are not actively tested)

Install
~~~~~~~
Install the latest version of virocon from PyPI by typing

.. code:: console

   pip install virocon


Alternatively, you can install from virocon repository’s Master branch
by typing

.. code:: console

   pip install https://github.com/virocon-organization/virocon/archive/master.zip
   
   
virocon is also available as a `conda`_ package. We recommend to first create a new environment.

.. code:: console

   conda create --name virocon python=3.9

And then activate that new environment and install virocon.

.. code:: console

   conda activate virocon
   conda install -c virocon-organization virocon


Usage
~~~~~

virocon is designed as an importable package.

The folder `examples`_ contains python files that show how one can
import and use virocon.

As an example, to run the file `hstz_contour_simple.py`_, use
your shell to navigate to the folder that contains the file. Make sure
that you have installed matplotlib and run the Python file by typing

.. code:: console

   python hstz_contour_simple.py

Documentation
-------------
**Learn.** Our `User Guide`_ covers installation, requirements and overall work flow.

**Code.** The code’s documentation can be found `here`_.

**Paper.** Our `SoftwareX paper`_ "ViroCon: A software to compute multivariate
extremes using the environmental contour method." provides a concise
description of virocon (version 1).

**Conference presentation.** In a `WESC 2021 presentation`_, we showed how virocon 
(version 2) can be used to support the design process of offshore wind turbines. 

Contributing
------------

**Issue.** If you spotted a bug, have an idea for an improvement or a
new feature, please open a issue. Please open an issue in both cases: If
you want to work on it yourself and if you want to leave it to us to
work on it.

**Fork.** If you want to work on an issue yourself please fork the
repository, then develop the feature in your copy of the repository and
finally file a pull request to merge it into our repository.

**Conventions.** We use PEP8.

License
-------

This software is licensed under the MIT license. For more information,
read the file `LICENSE`_.

.. _User Guide: https://virocon.readthedocs.io/en/latest/user_guide.html
.. _shell: https://en.wikipedia.org/wiki/Command-line_interface#Modern_usage_as_an_operating_system_shell
.. _www.python.org: https://www.python.org
.. _examples: https://github.com/virocon-organization/virocon/tree/master/examples
.. _hstz_contour_simple.py: https://github.com/virocon-organization/virocon/blob/master/examples/hstz_contour_simple.py
.. _here: https://virocon.readthedocs.io/en/latest/index.html
.. _LICENSE: https://github.com/virocon-organization/virocon/blob/master/LICENSE
.. _SoftwareX paper: https://github.com/ahaselsteiner/publications/blob/master/2018-10-25_SoftwareX_ViroCon_revised.pdf
.. _conda: https://docs.conda.io/en/latest/
.. _WESC 2021 presentation: http://doi.org/10.13140/RG.2.2.35455.53925
