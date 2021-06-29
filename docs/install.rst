************
Installation
************
Requirements
~~~~~~~~~~~~
Make sure you have installed Python `3.9` by typing

.. code:: console

   python --version

in your `shell`_.

(Older version might work, but are not actively tested)

Consider using the python version management `pyenv`_.


Install
~~~~~~~
Install the latest version of virocon from PyPI by typing

.. code:: console

   pip install virocon

Alternatively, you can install from virocon repository’s Master branch
by typing

.. code:: console

   pip install pip install https://github.com/virocon-organization/virocon/archive/master.zip

virocon is also available as a conda_ package. We recommend to first create a new environment.

.. code:: console

    conda create --name virocon python=3.9

And then activate that new environment and install virocon.

.. code:: console

    conda activate virocon
    conda install -c virocon-organization virocon

.. _shell: https://en.wikipedia.org/wiki/Command-line_interface#Modern_usage_as_an_operating_system_shell
.. _pyenv: https://github.com/pyenv/pyenv
.. _conda: https://docs.conda.io/en/latest/
