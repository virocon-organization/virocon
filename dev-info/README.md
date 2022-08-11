# Releasing new versions

When a new version is released, we make sure that it is available in GitHub archive, on PyPi and on anaconda.org.

GitHub archive:

* checkout master
* in virocon/__init__.py change __version__ = "1.1.3" (to the new version number)
* `git commit`
* `git tag 1.1.3 -m "Your commit message"`
* `git push --tags origin master`
* create a release from the tag on Github

PyPI:

* make sure you have build and twine installed:
    * `python3 -m pip install --upgrade build`
    * `python3 -m pip install --upgrade twine`
* OR use a conda env
    * `conda env create -f dev-info/virocon-pip-build.yml` (create build env, if not done yet)
    * `conda env update -f dev-info/virocon-pip-build.yml --prune` (update the build env if necessary)
    * `conda activate virocon-pip-build`
* `python3 -m build`
* `python3 -m twine upload --repository testpypi dist/*`
* `python3 -m twine upload dist/*`
* in case of errors/questions consult https://packaging.python.org/en/latest/tutorials/packaging-projects/#generating-distribution-archives

PyPI(old approach, uploads only a wheel):

* `python setup.py bdist_wheel`
* `twine upload dist/* -r pypitest`
* `twine upload dist/*`

Anaconda:

* `conda env create -f dev-info/virocon-build.yml` (create build env, if not done yet)
* `conda env update -f dev-info/virocon-build.yml --prune` (update the build env if necessary)
* `conda activate virocon-build`
* `conda build --python 3.9 conda-recipe`
* `anaconda login` (if not already logged in)
* `anaconda upload -u virocon-organization <path of file mentioned in build>`

 ## Generating HTML documentation on the local machine
 
 First make sure sphinx is installed:

* `conda install sphinx` or 
* `pip install sphinx`

There are two ways to produce the documentation of virocon on a local machine
 
First approach (Linux/ MacOS/ Windows cmd):

* `cd docs`
* `make clean`
* `make html`
 
-> Then open index.html located in docs/_build/html .

Second approach (Windows powershell): 

* `cd docs`
* `sphinx-build -b html . _build/html`
 
-> Then open index.html located in docs/_build/html .
