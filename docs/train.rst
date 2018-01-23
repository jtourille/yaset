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

* **general**: parameters related to the experiment (see below for further explanations)
* **data**: parameters related to training instances and word embedding models
* **training**: parameters related to model training (e.g. learning algorithm,
  evaluation metrics or mini-batch size)
* **<model parameters>**: depending on your choice regarding the neural
  network model to use (specified in the *training* section), you can
  modify the model parameters (e.g. hidden layer sizes or character
  embedding size)


general section
^^^^^^^^^^^^^^^

 ``batch_mode: bool``
  Set this parameter to ``true`` if you want to perform multiple runs of
  the same experiment. This allows to check the model robustness to random
  seed initial value (Reimers et al. (2017) :cite:`Reimers2017`).

 ``batch_iter: int``
  Specify the number of runs to perform. This will be ignored if the value
  of the parameter ``batch_mode`` is ``false``

 ``experiment_name: str``
  Specify the experiment name. The name will be used for directory and file
  naming.

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

training
^^^^^^^^

 ``model_type: str``
  Specify the neural network model that you want to use. There is only one
  choice at this time. Other models will be implemented in the next releases.
   * ``bilstm-char-crf``: implementation of the model presented in
     Lample et al. (2016) :cite:`Lample2016`. More information can be found
     in the original paper. Model parameters can be set in the
     :ref:`bilstm-char-crf section <bilstm-char-crf>` of the configuration
     file.

 ``max_iterations: int``
  Specify the maximum number of training iterations. Training will be stopped
  if early stopping criterion is not reached before this iteration number (see
  ``patience`` parameter).

 ``patience: int``
  Specify the number of iterations to wait before early stop if there is no
  performance improvement on the validation instances.

 ``dev_metric: str``
  Specify the metric used for performance computation on the validation
  instances.
   * ``accuracy``: standard token accuracy.
   * ``conll``: metric which operates at the entity level. This
     should be used with a IOB(ES) markup on Named Entity Recognition related
     tasks. The implementation is taken for most parts from the
     `Python adaptation`_ by Sampo Pyysalo of the original script developed
     for the
     `CoNLL-2003 Shared Task`_ (Tjong et al., 2003 :cite:`TjongKimSang2003`).

 ``trainable_word_embeddings: bool``
  Set this parameter to ``true`` if you want YASET to fine-tune word
  embeddings during network training, ``false`` otherwise.

 ``cpu_cores: int``
  Specify the number of CPU cores (upper-bound) that should be used during
  network training.

 ``batch_size: int``
  Specify the mini-batch size used during training.

 ``store_matrices_on_gpu: bool``
  Set this parameter to ``true`` if you want to keep the word embedding matrix
  on GPU memory, ``false`` otherwise.

 ``bucket_use: bool``
  Set this parameter to ``true`` if you want to bucketize training instances
  during network training. Bucket boundaries will be automatically computed.

 ``opt_algo: str``
  Specify the optimization algorithm used during network training. You can
  choose between between ``adam`` (Kingma et al.,2014 :cite:`Kingma2015`)
  or ``sgd``.

 ``opt_lr: float``
  Specify the initial learning rate applied during network training.

 ``opt_gc_use: bool``
  Set this parameter to ``true`` if you want to use gradient clipping during
  network training, ``false`` otherwise.

 ``opt_gc_type: str``
  Specify the gradient clipping type (``clip_by_norm`` or ``clip_by_value``)
  This will be ignored if the value of the parameter ``opt_gc_use`` is
  ``false``.

 ``opt_gs_val: float``
  Specify the gradient clipping value. This parameter will be ignored if the
  value for the parameter ``opt_gc_use`` is ``false``.

 ``opt_decay_use: bool``
  Set this parameter to ``true`` if you want to use learning rate decay during
  network training, ``false`` otherwise.

 ``opt_decay_rate: float``
  Specify the decay rate (float between 0 and 1, e.g. 0.2). This parameter
  will be ignored if the value for the parameter ``opt_decay_use`` is
  ``false``.

 ``opt_decay_iteration: int``
  Specify the learning rate decay frequency. If you set the frequency to
  :math:`n`, the learning rate :math:`lr` will be decayed by the rate
  specified in the parameter ``opt_decay_iteration`` every :math:`n`
  iterations.

.. _bilstm-char-crf:

bilstm-char-crf
^^^^^^^^^^^^^^^
These parameters are related to the neural network model presented in
Lample et al. (2016) :cite:`Lample2016`.

 ``hidden_layer_size: int``
  Specify the main LSTM hidden layer size.

 ``dropout_rate: float``
  Specify the dropout rate to apply on input embeddings before feeding them
  to the main LSTM.

 ``use_char_embeddings: bool``
  Set this parameter to ``true`` if you want to use character embeddings in
  the model, ``false`` otherwise.

 ``char_hidden_layer_size: int``
  Specify the character LSTM hidden layer size. This parameter will be ignored
  if the value for the parameter ``use_char_embeddings`` is ``false``.

 ``char_embedding_size: int``
  Specify the character embedding size. This parameter will be ignored
  if the value for the parameter ``use_char_embeddings`` is ``false``.

.. _gensim: https://radimrehurek.com/gensim/
.. _word2vec: https://github.com/dav/word2vec
.. _Python adaptation: https://github.com/spyysalo/conlleval.py
.. _CoNLL-2003 Shared Task: https://www.clips.uantwerpen.be/conll2003/ner/
.. bibliography:: refs.bib
   :filter: docname in docnames
   :style: plain
