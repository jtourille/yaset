Getting Started
===============

This document will show you how to install and upgrade YASET.

Requirements
------------

* You need a working **Python 3.3+** environment.
* Optional: we recommend the use of Graphics Processing Units (GPU).
  YASET implements state-of-the-art neural network models for
  sequence tagging. Hence, it can leverage GPU computational power to
  speed up computation. However, this requirement is optional and the tool
  should be able to learn models in a reasonable time frame using only
  Central Processing Units (CPU).


Installation
------------

Here are the steps to follow in order to install YASET.

1. **Download** the `latest stable version`_ on GitHub or **clone the repository** if you want to use the cutting-edge development version.

::

	git clone git@github.com:jtourille/yaset.git


3. Uncompress the file if necessary and move to the newly newly created directory. Install YASET by invoking `pip`.

::

    cd yaset
    pip install .


GPU Support
-----------

YASET install the non-GPU version of TensorFlow by default. If you want to use the GPU version, upgrade the TensorFlow package.

::

    pip install tensorflow-gpu==1.2.0


Upgrade
-------

If you want to upgrade to a newer version, download the last release or pull the last version of the repository and then
upgrade the package. If you switched to the GPU version of TensorFlow, the change will be kept. You do not need to repeat
TensorFlow upgrade.

::

	git pull
	pip install --upgrade .

.. _latest stable version: https://github.com/jtourille/yaset/releases/latest