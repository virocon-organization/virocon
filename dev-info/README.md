# Releasing new versions

When a new version is released, we make sure that it is available in GitHub archive, on PyPi and on anaconda.org.

GitHub archive:
* checkout master
* in virocon/__init__.py change __version__ = "1.1.3" (to the new version number)
* `git commit`
* `git tag 1.1.3 -m "Your commit message"`
* `git push --tags origin master`

PyPI:
* `python setup.py bdist_wheel`
* `twine upload dist/* -r pypitest`
* `twine upload dist/*`

Anaconda:
* `conda env create -f dev-info/conda_build_env.yml` (create build env, if not done yet)
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
