virocon
=======

.. image:: https://travis-ci.org/virocon-organization/viroconcom.svg?branch=master
    :target: https://travis-ci.org/virocon-organization/viroconcom
    :alt: Build status

.. image:: https://coveralls.io/repos/github/virocon-organization/viroconcom/badge.svg?branch=master
    :target: https://coveralls.io/github/virocon-organization/viroconcom?branch=master

virocon is a software to compute environmental contours. `User Guide`_

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

   pip install https://github.com/virocon-organization/viroconcom/archive/master.zip


Usage
~~~~~

virocon is designed as an importable package.

The folder `examples`_ contains python files that show how one can
import and use virocon.

As an example, to run the file `sea_state_iform_contour.py`_, use
your shell to navigate to the folder that contains the file. Make sure
that you have installed matplotlib and run the Python file by typing

.. code:: console

   python sea_state_iform_contour.py

Documentation
-------------
**Learn.** Our `User Guide`_ covers installation, requirements and overall work flow.

**Code.** The code’s documentation can be found `here`_.

**Paper.** Our `SoftwareX paper`_ "ViroCon: A software to compute multivariate
extremes using the environmental contour method." provides a concise
description of the software.

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

.. _User Guide: https://virocon-organization.github.io/virocon/user_guide.html
.. _shell: https://en.wikipedia.org/wiki/Command-line_interface#Modern_usage_as_an_operating_system_shell
.. _www.python.org: https://www.python.org
.. _examples: https://github.com/virocon-organization/viroconcom/tree/master/examples
.. _sea_state_iform_contour.py: https://github.com/virocon-organization/viroconcom/blob/master/examples/sea_state_iform_contour.py
.. _here: https://virocon-organization.github.io/viroconcom/
.. _LICENSE: https://github.com/virocon-organization/viroconcom/blob/master/LICENSE
.. _SoftwareX paper: https://github.com/ahaselsteiner/publications/blob/master/2018-10-25_SoftwareX_ViroCon_revised.pdf
