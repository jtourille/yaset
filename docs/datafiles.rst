Data
====

.. _data-formatting:

Training instances
------------------

YASET accepts CoNLL-like formatted data:

* One token per line
* Sequences separated by blank lines

The first column must contain the tokens and the last column must contain the
labels. You can add as many other columns as you need, they will be ignored
by the system. Columns must be separated by tabulations.

To illustrate this format, here is an example extracted from the English
part of the `CoNLL-2003 Shared Task corpus`_ :cite:`TjongKimSang2003`.

::

    ...

    EU	NNP	I-NP	I-ORG
    rejects	VBZ	I-VP	O
    German	JJ	I-NP	I-MISC
    call	NN	I-NP	O
    to	TO	I-VP	O
    boycott	VB	I-VP	O
    British	JJ	I-NP	I-MISC
    lamb	NN	I-NP	O
    .	.	O	O

    ...

Word Embeddings
---------------

YASET supports two types of word embeddings:

* `Gensim`_ models
* word2vec models (binary or text)

If you want to use other types of embeddings, you must first convert them to one of these two formats. For instance, if
you have computed word embeddings with `Glove`_ :cite:`Pennington2014`, you can convert the file to word2vec text format by using the `script`_ provided within the gensim library.


.. bibliography:: refs.bib
.. _CoNLL-2003 Shared Task corpus: https://www.clips.uantwerpen.be/conll2003/ner/
.. _Gensim: https://radimrehurek.com/gensim/
.. _script: https://radimrehurek.com/gensim/scripts/glove2word2vec.html
.. _glove: https://nlp.stanford.edu/projects/glove/