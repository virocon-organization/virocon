# Releasing new versions

When a new version is released, we make sure that it is available in GitHub archive, on PyPi and on conda-forge.

GitHub archive:

* checkout main
* in virocon/__init__.py change __version__ = "1.2.3" (to the new version number)
* `git commit -am "Bump version to 1.2.3"`
* `git tag -a 1.2.3 -m "Release version 1.2.3"`
* `git push --tags origin main`
* create a release from the tag on Github

PyPI:

* testpypi
  * `uv publish --index testpypi`
* pypi
  * `uv publish`
* The commands will ask for username and password.
  Use `__token__` as username and your [access token](https://pypi.org/help/#apitoken) as password

conda-forge:

* Update the version in https://github.com/conda-forge/virocon-feedstock.
  * Either create and merge PR as described in https://github.com/conda-forge/virocon-feedstock?tab=readme-ov-file#updating-virocon-feedstock
  * or wait for a bot to create that PR and merge it.

 ## Generating HTML documentation on the local machine

There are two ways to produce the documentation of virocon on a local machine

First approach (Linux/ MacOS/ Windows cmd):

* `cd docs`
* `uv run make clean`
* `uv run make html`

-> Then open index.html located in docs/_build/html .

Second approach (Windows powershell):

* `cd docs`
* `uv run sphinx-build -b html . _build/html`

-> Then open index.html located in docs/_build/html .

# Install virocon from a feature branch

If you like to install a feature branch virocon version during the development run

`pip install https://github.com/virocon-organization/virocon/archive/feature-branch.zip`

where `feature-branch`is the name of your feature branch.
