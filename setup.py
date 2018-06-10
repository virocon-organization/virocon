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

with open(path.join(here, 'viroconcom/VERSION')) as version_file:
    version = version_file.read().strip()


setup(
    name='viroconcom',

    version=version,  # Required

    description='ViroCon\'s numerical core: computes n-dimensional environmental contours',

    long_description=long_description,

    url='https://github.com/ahaselsteiner/viroconcom',

    # We use git tags for this download_url. This approach is based on:
    # https://peterdowns.com/posts/first-time-with-pypi.html
    download_url = 'https://github.com/ahaselsteiner/viroconcom/archive/' + version + '.tar.gz',

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

    # This field adds keywords for your project which will appear on the
    # project page. What does your project relate to?
    #
    keywords='virocon environmental contours',


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
        'Documentation': 'https://ahaselsteiner.github.io/viroconcom/',
        'Source Code': 'https://github.com/ahaselsteiner/viroconcom',
    },
)
