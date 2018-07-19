ViroCon: viroconcom
===================

.. image:: https://travis-ci.org/virocon-organization/viroconcom.svg?branch=master
    :target: https://travis-ci.org/virocon-organization/viroconcom
    :alt: Build status

.. image:: https://coveralls.io/repos/github/virocon-organization/viroconcom/badge.svg?branch=master
    :target: https://coveralls.io/github/virocon-organization/viroconcom?branch=master

ViroCon is a software to compute environmental contours.

About
-----

viroconcom is a package belonging to the software ViroCon. The package viroconcom
handles the statistical computations.

ViroCon helps you to design marine structures, which need to withstand
load combinations based on wave, wind and current. It lets you define
extreme environmental conditions with a given return period using the
environmental contour method.

The following methods are implemented in viroconcom:

- Fitting a probabilistic model to measurement data using maximum likelihood estimation
- Defining a probabilistic model with the conditonal modeling approach (CMA)
- Computing an environmental contour using either the

  - inverse first-order reliability method (IFORM),
  - inverse second-order reliability method (ISORM) or the
  - highest density contour (HDC) method


ViroCon is written in Python 3.6.4. The software is seperated in two
main packages, viroconweb and viroconcom. This is the repository of
viroconcom, which is the numerical core. It handles the statistical
computations. viroconweb builds on viroconcom and is a web-based
application with a graphical user interface. It has its own
`repository`_.

How to use viroconcom
---------------------
Requirements
~~~~~~~~~~~~
Make sure you have installed Python `3.6.4`_ by typing

.. code:: console

   python --version

in your `shell`_.

Consider using the python version management `pyenv`_.


Install
~~~~~~~
Install the latest version of viroconcom from PyPI by typing

.. code:: console

   pip install viroconcom

in your shell.

Alternatively, you can install from viroconcom repository’s Master branch
by typing

.. code:: console

   pip install https://github.com/virocon-organization/viroconcom/archive/master.zip

in your shell.

Usage
~~~~~

viroconcom is designed as an importable package.

The documentation gives examples how to `fit a distribution`_ to measurement data
and how to `compute environmental contours`_.

Additionally, the folder `examples`_ contains python files that show how one can
import and use viroconcom.

As an example, to run the file `calculate_contours_similar_to_docs.py`_, use
your shell to navigate to the folder that contains the file. Then make sure
that you have installed the python package matplotlib or install it by typing

.. code:: console

   pip install matplotlib

in your shell.

Now run the Python file by typing

.. code:: console

   python calculate_contours_similar_to_docs.py

in your shell.

Documentation
-------------

**Code.** The code’s documentation can be found `here`_.

**Paper.** We are currently writing an academic paper describing
ViroCon. We will provide a link to it here.

Contributing
------------

There are various ways you can contribute. You could

- improve the code,
- improve the documentation,
- add a feature or
- report a bug or an improvement and leave it to us to implement it.

**Issue.** If you spotted a bug, have an idea for an improvement or a
new feature, please open a issue. Please open an issue in both cases: If
you want to work on in yourself and if you want to leave it to us to
work on it.

**Fork.** If you want to work on an issue yourself please fork the
repository, then develop the feature in your copy of the repository and
finally file a pull request to merge it into our repository.

**Conventions.** In our `Contribution Guide`_ we summarize our
conventions, which are consistent with PEP8.

License
-------

This software is licensed under the MIT license. For more information,
read the file `LICENSE`_.

.. _repository: https://github.com/virocon-organization/viroconweb
.. _3.6.4: https://www.python.org/downloads/release/python-364
.. _shell: https://en.wikipedia.org/wiki/Command-line_interface#Modern_usage_as_an_operating_system_shell
.. _pyenv: https://github.com/pyenv/pyenv
.. _www.python.org: https://www.python.org
.. _fit a distribution: https://virocon-organization.github.io/viroconcom/fitting.html
.. _compute environmental contours: https://virocon-organization.github.io/viroconcom/contours.html
.. _examples: https://github.com/virocon-organization/viroconcom/tree/master/examples
.. _calculate_contours_similar_to_docs.py: https://github.com/virocon-organization/viroconcom/blob/master/examples/calculate_contours_similar_to_docs.py
.. _here: https://virocon-organization.github.io/viroconcom/
.. _Contribution Guide: https://virocon-organization.github.io/viroconcom/contributionguide.html
.. _LICENSE: https://github.com/virocon-organization/viroconcom/blob/master/LICENSE
