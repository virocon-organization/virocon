"""Virocon's numerical core
"""

# This is based on the setup.py example at:
# https://github.com/pypa/sampleproject/blob/master/setup.py

from setuptools import setup
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

version = {}
with open(path.join(here, 'viroconcom/version.py')) as version_file:
    exec(version_file.read(), version)

setup(
    name='viroconcom',

    version=version['__version__'],

    description='ViroCon\'s numerical core: computes n-dimensional environmental contours',

    long_description=long_description,

    url='https://github.com/virocon-organization/viroconcom',

    # We use git tags for this download_url. This approach is based on:
    # https://peterdowns.com/posts/first-time-with-pypi.html
    download_url = 'https://github.com/virocon-organization/viroconcom/archive/' + version['__version__'] + '.tar.gz',

    author='ViroCon Team',

    author_email='virocon@uni-bremen.de',


    # For a list of valid classifiers, see
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3 :: Only',
    ],

    packages=['viroconcom'],

    install_requires=[
            'numpy',
            'scipy',
            'statsmodels'],


    extras_require={
        'dev': ['sphinx'],
        'test': [],
    },

    # This field corresponds to the "Project-URL" metadata fields:
    # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    project_urls={  # Optional
        'Documentation': 'https://virocon-organization.github.io/viroconcom/',
        'Source Code': 'https://github.com/virocon-organization/viroconcom',
    },
)
