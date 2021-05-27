# Releasing new versions

When a new version is released, we make sure that it is available in GitHub archive, on PyPi and on anaconda.org.

GitHub archive:
* checkout master
* git commit
* git tag 1.1.3 -m "Your commit message"
* git push --tags origin master

PyPI:
 * python setup.py bdist_wheel
 * twine upload dist/* -r pypitest 
 * twine upload dist/*

Anaconda (commands in parantheses are only needed once before the first upload):
 * (conda install conda-build)
 * (conda install anaconda-client)
 * conda build (--python 3.9) conda-recipe
 * anaconda login
 * anaconda upload -u virocon-organization <path of file mentioned in build>
