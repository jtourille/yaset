Train a model
=============

Quick Start
-----------

To train a model, first make a copy of the configuration sample and adjust the
parameters to you situation

::

	cp config.ini config-xp.ini

Invoke the yaset command:

::

	yaset LEARN --config config-xp.ini

Configuration Parameters
------------------------

The configuration file is divided into 3 parts:

* **data**: parameters related to training instances and word embedding models
* **training**: parameters related to model training (e.g. learning algorithm,
  metrics or mini-batch size)
* **<model parameters>**: depending on your choice regarding the neural
  network model to use (specified in the *training* section), you can
  modify the model parameters (e.g. hidden layer sizes or character
  embedding size)


data section
^^^^^^^^^^^^

 train_file_path
  Specify the *training instance file path* (absolute or relative).
  Please refer to the :ref:`data formatting <data-formatting>` section for
  further information about file format.

 dev_file_use
  Set this parameter to *true* if you want to use a development instance file,
  *false* otherwise.

 dev_file_path
  Specify the *development instance file path* if you set the previous
  parameter (*dev_file_use*) to *true*. It can be absolute or relative.
  Please refer to the :ref:`data formatting <data-formatting>` section
  for further information about file format.

 dev_random_ratio
  Specify the percentage training instances that should be kept as development
  instances (float between 0 and 1, e.g. 0.2). This will be ignored if the
  value of the parameter *dev_file_use* is *true*.

 dev_random_seed_use
  Set this parameter to *true* if you want to use a random seed for train/dev
  split. This will be ignored if the value of the parameter *dev_file_use* is
  *true*.

 dev_random_seed_value
  Specify the random seed value (integer). This will be ignored if the value
  of the parameter *dev_file_use* is *true* or if the value of the parameter
  *dev_random_seed_use* is *false*

 preproc_lower_input
  Set this parameter to *true* if you want YASET to lowercase tokens before
  token-vector looking-up, *false* otherwise. This is useful if you
  pre-trained word embeddings using a lowercased corpus.

 preproc_replace_digits
  Set this parameter to *true* if you want YASET to replace digits by the
  digit *0* before token-vector looking-up, *false* otherwise (e.g. "4,5mg"
  will be changed to "0,0mg").

 embedding_model_type
  Specify the format of the pre-trained word embeddings that you want to use
  to train the system. Two formats are supported:

   * *gensim*: models pre-trained with the Python library `gensim`_.
   * *word2vec*: models pre-trained with the tool `word2vec`_.

 embedding_model_path
  Specify the path of the pre-trained word embedding file (absolute or
  relative)

 embedding_oov_strategy
  Specify the strategy for Out-Of-Vocabulary (OOV) tokens. Two strategies are
  available:

  * **map**: you have a vector for OOV tokens in the embedding file you
    provided. Set *embedding_oov_strategy* to *map* and specify the OOV
    vector ID (*embedding_oov_map_token_id* parameter)
  * **replace**: following Lample et al. (2016) :cite:`Lample2016`, an OOV
    vector will be randomly initialized and trained by randomly replacing
    singletons in the training instances by this vector. You can adjust the
    replacement rate by changing the value of the parameter
    *embedding_oov_replace_rate*.

 embedding_oov_map_token_id
  Specify the *unknown* token ID if you use the strategy *map*. This will be
  ignored if the value of the parameter *embedding_oov_strategy* is not *map*.

 embedding_oov_replace_rate
  Specify the replacement rate if you want to use the strategy *replace*. This
  will be ignored if the value of the parameter *embedding_oov_strategy* is
  not *replace*.

.. _gensim: https://radimrehurek.com/gensim/
.. _word2vec: https://github.com/dav/word2vec
.. bibliography:: refs.bib
   :filter: docname in docnames
