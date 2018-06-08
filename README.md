# viroconcom

[![Build Status](https://travis-ci.org/ahaselsteiner/viroconcom.svg?branch=master)](https://travis-ci.org/ahaselsteiner/viroconcom)
[![Coverage Status](https://coveralls.io/repos/github/ahaselsteiner/viroconcom/badge.svg?branch=master)](https://coveralls.io/github/ahaselsteiner/viroconcom?branch=master)

## About
viroconcom is a package belonging to the software ViroCon.

ViroCon helps you to design marine structures, which need to withstand load
combinations based on wave, wind and current. It lets you define extreme
environmental conditions with a given return period using the environmental
contour method.

The following methods are available:
* Fitting a probabilistic model to measurement data using maximum likelihood
estimation
* Defining a probabilistic model with the conditonal modeling approach (CMA)
* Computing an environmental contour using either the
  * inverse first order reliability method (IFORM) or the
  * highest density contour (HDC) method

ViroCon is written in Python 3.6.4. The software is seperated in two main
packages, viroconweb and viroconcom. This is the repository of viroconcom,
which is the numerical core. It handles the statistical computations. viroconweb
 builds on viroconcom and is a a web-based application with a graphical user
 interface. It has its own [repository](https://github.com/ahaselsteiner/viroconweb).

## Install
To get the latest version from PyPI use
```console
pip install viroconcom
```

Alternatively you can install from viroconcom repository's Master branch with
```console
pip install https://github.com/ahaselsteiner/viroconcom/archive/master.zip
```

## Documentation
**Code.** The code's documentation can be found
[here](https://ahaselsteiner.github.io/viroconcom/).

**Paper.** We are currently writing an academic paper describing ViroCon. We will
provide a link to it here.

## Contributing
There are various ways you can contribute. You could
 * improve the code,
 * improve the documentation,
 * add a feature or
 * report a bug or an improvement and leave it to us to implement it.

**Issue.** If you spotted a bug, have an idea for an improvement or a new
 feature please open a issue. Please open an issue in both cases: If you want to
 work on in yourself and if you want to leave it to us to work on it.

**Fork.** If you want to work on an issue yourself please fork the repository,
then develop the feature in your copy of the repository and finally
file a pull request to merge it into our repository.

**Conventions.** In our [Contribution Guide](https://ahaselsteiner.github.io/viroconcom/styleguide.html)
we summarize our conventions, which are consistent with PEP8.

## License
This software is licensed under the MIT license. For more information, read the
file [LICENSE](https://github.com/ahaselsteiner/viroconcom/blob/master/LICENSE).
