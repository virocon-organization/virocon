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


    classifiers=[ 
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
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

    project_urls={  # Optional
        'Documentation': 'https://virocon-organization.github.io/viroconcom/',
        'Source Code': 'https://github.com/virocon-organization/viroconcom',
    },
)
