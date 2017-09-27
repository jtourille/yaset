Getting Started
===============

This document will show you how to install `yaset`, train and apply a model

Installation
------------

Here are the steps to follow in order to install `yaset`.

1. To install `yaset`, you need a working **Python 3.5+** environment.


2. You can either **download** the `latest stable version`_ on GitHub or **clone the repository** if you want the latest development version.

::
	
	git clone git@github.com:jtourille/yaset.git


3. Install `yaset` by invoking `pip`.

::
	
	pip install .

Prepare the data
----------------

`yaset` accepts CoNLL-like formatted data:

* One token per line
* Sequences separated by blank lines

The first column must contain tokens and the last column must contain the labels. You can add as many other columns as you wish, they will be ignored by the system. Columns must be separated by tabulations.

Train a model
-------------

To train a model, first make a copy of the configuration sample and adjust the parameters to you situation

::

	cp config.ini config-xp.ini

Invoke the yaset command:

::

	yaset LEARN --config config-xp.ini


Apply a model
-------------

To apply a model, run the following command

::

	yaset APPLY --working_dir /path/to/working_dir \
		--input_file /path/to/file.tab \
		--model_path /path/to/pretrained_model

.. _latest stable version: https://github.com/jtourille/yaset/releases/latest