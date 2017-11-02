Train a model
=============

Quick Start
-----------

To train a model, first make a copy of the configuration sample and adjust the parameters to you situation

::

	cp config.ini config-xp.ini

Invoke the yaset command:

::

	yaset LEARN --config config-xp.ini

Configuration Parameters
------------------------

The configuration file is divided into 3 parts:

* **data**: parameters related to training instances and word embedding models
* **training**: parameters related to model training (e.g. learning algorithm, metrics or mini-batch size)
* **<model parameters>**: depending on your choice regarding the neural network
  model to use (specified in the *training* section), you can modify the model
  parameters (e.g. hidden layer sizes or character embedding size)


data section
^^^^^^^^^^^^

 train_file_path
  Specify the *training instance file path*. It can be absolute or relative.
  Please refer to the ":ref:`data-formatting`" section for further information about file format.

 dev_file_use
  Set this parameter to *true* if you want to use a development instance file, *false* otherwise.

 dev_file_path
  Specify the *development instance file path* if you set the previous parameter to *true*.
  It can be absolute or relative. Please refer to the ":ref:`data-formatting`" section
  for further information about file format.

 dev_random_ratio
  Specify the percentage of the training instances that should be kept as development
  instances (float between 0 and 1, e.g. 0.2). This will be ignored if the value of
  the parameter *dev_file_use* is *true*.



