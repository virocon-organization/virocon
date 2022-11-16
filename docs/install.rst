************
Installation
************
Requirements
~~~~~~~~~~~~
Make sure you have installed Python `3.8` or `3.9` by typing

.. code:: console

   python --version

in your `shell`_.

(Older version might work, but are not actively tested)

If you do not (want to) use conda, consider using a python version management like pyenv_ .


Install
~~~~~~~
Install the latest version of virocon from PyPI by typing

.. code:: console

   pip install virocon


Alternatively, you can install from virocon repository’s Master branch
by typing

.. code:: console

   pip install https://github.com/virocon-organization/virocon/archive/master.zip
   
   
virocon is also available on `conda-forge`_. We recommend to first create a new environment.

.. code:: console

   conda create --name virocon python=3.10

And then activate that new environment and install virocon.

.. code:: console

   conda activate virocon
   conda install -c conda-forge virocon


.. _shell: https://en.wikipedia.org/wiki/Command-line_interface#Modern_usage_as_an_operating_system_shell
.. _pyenv: https://github.com/pyenv/pyenv
.. _conda-forge: https://conda-forge.org/
