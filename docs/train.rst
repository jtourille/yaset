Train a model
=============

This document explains how to train a model with YASET.

Quick Start
-----------

To train a model, make a copy of the configuration file sample and adjust the
parameters to you situation.

.. code-block:: shell

	$ cp config.ini config-xp.ini

Invoke the YASET command. You can turn on the *verbose mode* with the debug
flag (``--debug``).

.. code-block:: shell

	$ yaset [--debug] LEARN --config config-xp.ini

Configuration Parameters
------------------------

The configuration file is divided into 3 parts:

* **data**: parameters related to training instances and word embedding models
* **training**: parameters related to model training (e.g. learning algorithm,
  evaluation metrics or mini-batch size)
* **<model parameters>**: depending on your choice regarding the neural
  network model to use (specified in the *training* section), you can
  modify the model parameters (e.g. hidden layer sizes or character
  embedding size)


data section
^^^^^^^^^^^^

 ``train_file_path: str``
  Specify the *training instance file path* (absolute or relative).
  Please refer to the :ref:`data formatting section <data-formatting>`
  of the :doc:`data document <datafiles>` for further information about file
  format.

 ``dev_file_use: bool``
  Set this parameter to ``true`` if you want to use a development instance
  file, ``false`` otherwise.

 ``dev_file_path: str``
  Specify the *development instance file path* (absolute or relative).
  This parameter will be ignored if the value of the parameter
  ``dev_file_use`` is set to ``false``. Please refer to the
  :ref:`data formatting section <data-formatting>` of the
  :doc:`data document <datafiles>` for further information about file
  format.

 ``dev_random_ratio: float``
  Specify the percentage of training instances that should be kept as
  development instances (float between 0 and 1, e.g. 0.2). This will
  be ignored if the value of the parameter ``dev_file_use`` is ``true``.

 ``dev_random_seed_use: bool``
  Set this parameter to ``true`` if you want to use a random seed for
  train/dev split. This will be ignored if the value of the parameter
  ``dev_file_use`` is ``true``.

 ``dev_random_seed_value: int``
  Specify the random seed value (integer). This will be ignored if the value
  of the parameter ``dev_file_use`` is ``true`` or if the value of the
  parameter ``dev_random_seed_use`` is ``false``

 ``preproc_lower_input: bool``
  Set this parameter to ``true`` if you want YASET to lowercase tokens before
  token-vector looking-up, ``false`` otherwise. This is useful if you
  have pre-trained word embeddings using a lowercased corpus.

 ``preproc_replace_digits: bool``
  Set this parameter to ``true`` if you want YASET to replace digits by the
  digit 0 before token-vector looking-up, ``false`` otherwise (e.g. "4,5mg"
  will be changed to "0,0mg").

 ``embedding_model_type: str``
  Specify the format of the pre-trained word embeddings that you want to use
  to train the system. Two formats are supported:

   * ``gensim``: models pre-trained with the Python library `gensim`_
   * ``word2vec``: models pre-trained with the tool `word2vec`_

 ``embedding_model_path: str``
  Specify the path of the pre-trained word embedding file (absolute or
  relative).

 ``embedding_oov_strategy: str``
  Specify the strategy for Out-Of-Vocabulary (OOV) tokens. Two strategies are
  available:

  * ``map``: a vector for OOV tokens is provided in the embedding file.
    Set *embedding_oov_strategy* to ``map`` and specify the OOV
    vector ID (``embedding_oov_map_token_id`` parameter)
  * ``replace``: following Lample et al. (2016) :cite:`Lample2016`, an OOV
    vector will be randomly initialized and trained by randomly replacing
    singletons in the training instances by this vector. You can adjust the
    replacement rate by changing the value of the parameter
    ``embedding_oov_replace_rate``.

 ``embedding_oov_map_token_id: str``
  Specify the OOV token ID if you use the strategy ``map``. This will be
  ignored if the value of the parameter ``embedding_oov_strategy`` is not
  ``map``.

 ``embedding_oov_replace_rate: float``
  Specify the replacement rate if you want to use the strategy ``replace``
  (float between 0 and 1, e.g. 0.2). This will be ignored if the value
  of the parameter ``embedding_oov_strategy`` is not ``replace``.

 ``working_dir: str``
  Specify the working directory path where a timestamped working directory
  will be created for the current run. For instance, if you specify
  ``$USER/temp``, the directory ``$USER/temp/yaset-learn-YYYYMMDD`` will be
  created.

.. _gensim: https://radimrehurek.com/gensim/
.. _word2vec: https://github.com/dav/word2vec
.. bibliography:: refs.bib
   :filter: docname in docnames
   :style: plain
