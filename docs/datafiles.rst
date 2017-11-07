Data
====

This document presents input file formatting requirements for train and test
instances, and word embeddings.

.. _data-formatting:

Train and Test Instances
------------------------

YASET accepts *CoNLL*-like formatted data:

* one token per line
* sequences separated by blank lines

The first column *\*must\** contain the tokens and the last column *\*must\**
contain the labels. You can add as many other columns as you need, they will
be ignored by the system. Columns *\*must*\* be separated by **tabulations**.

The example below which is extracted from the English part of the
`CoNLL-2003 Shared Task`_ corpus (Tjong et al., 2003
:cite:`TjongKimSang2003`) illustrates this format.

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

YASET supports two word embedding formats:

* `gensim`_ models (Řehůřek et al., 2010 :cite:`vRehruvrek2010`)
* `word2vec`_ models (Mikolov et al., 2013 :cite:`Mikolov2013`)

If you want to use other types of embeddings, you must first convert them to
one of these two formats. For instance, if you have computed word embeddings
with `Glove`_ (Pennington et al., 2014 :cite:`Pennington2014`), you can
convert the file to word2vec text format by using the `script`_ provided
within the gensim library.

.. rubric:: References

.. bibliography:: refs.bib
   :filter: docname in docnames
   :style: plain

.. _CoNLL-2003 Shared Task: https://www.clips.uantwerpen.be/conll2003/ner/
.. _gensim: https://radimrehurek.com/gensim/
.. _script: https://radimrehurek.com/gensim/scripts/glove2word2vec.html
.. _glove: https://nlp.stanford.edu/projects/glove/
.. _word2vec: https://github.com/dav/word2vec