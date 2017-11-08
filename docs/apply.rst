Apply a model
=============

This document explains how to apply a YASET model on test data.

Data Format
-----------

YASET accepts *CoNLL*-like formatted data:

* one token per line
* sequences separated by blank lines

The difference with train and validation data is that
test data do not need to have a label column. Hence the minimum number of
column is one (i.e. the token column). You can add as many other columns
you need. They will be ignored by the system.

.. code-block:: text

    ...

    EU	NNP	I-NP
    rejects	VBZ	I-VP
    German	JJ	I-NP
    call	NN	I-NP
    to	TO	I-VP
    boycott	VB	I-VP
    British	JJ	I-NP
    lamb	NN	I-NP
    .	.	O

    ...

During the APPLY phase, YASET will add one column to the document with
the predicted labels.

Apply a model
-------------

To apply a model, run the following command

.. code-block:: bash

   $ yaset [--debug] APPLY --working-dir /path/to/working_dir \
      --input-file /path/to/file.tab \
      --model-path /path/to/pre-trained-model

Argument description:

 ``--working-dir``
  Specify the working directory path where a timestamped working
  directory will be created for the current run. For instance,
  if you specify ``$USER/temp``, the directory
  ``$USER/temp/yaset-apply-YYYYMMDD`` will be created.

 ``--input-file``
  Specify the input test file which contains the test instances.

 ``--model-path``
  Specify the path of the YASET model

