# This is based on the setup.py example at:
# https://github.com/pypa/sampleproject/blob/master/setup.py

from setuptools import setup

# To use a consistent encoding
import codecs
import os.path


long_description = """
virocon
=======

virocon is a software to compute environmental contours.

A longer description is available at: https://github.com/virocon-organization/viroconcom
"""


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="virocon",
    version=get_version("virocon/__init__.py"),
    description="ViroCon is a software to compute environmental contours",
    long_description=long_description,
    url="https://github.com/virocon-organization/viroconcom",
    # We use git tags for this download_url. This approach is based on:
    # https://peterdowns.com/posts/first-time-with-pypi.html
    download_url="https://github.com/virocon-organization/viroconcom/archive/"
    + get_version("virocon/__init__.py")
    + ".tar.gz",
    author="virocon team",
    author_email="virocon@uni-bremen.de",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.7",
    packages=["virocon"],
    install_requires=[
        "matplotlib>=2.2.0" "networkx",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
    ],
    extras_require={
        "dev": ["Sphinx"],
        "test": ["pytest", "pytest-cov", "coverage", "coveralls"],
    },
    project_urls={  # Optional
        "Documentation": "https://virocon-organization.github.io/viroconcom/",
        "Source Code": "https://github.com/virocon-organization/viroconcom",
    },
)
