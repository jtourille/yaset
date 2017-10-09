import logging

import gensim
import numpy as np

from .embeddings import Embeddings


class Word2VecEmbeddings(Embeddings):

    def load_embedding(self):
        """
        Load embedding matrix and word count from gensim object
        :return: nothing
        """

        # Loading gensim object
        logging.debug("-> Loading word2vec file")
        try:
            gensim_obj = gensim.models.KeyedVectors.load_word2vec_format(self.embedding_file_path)
        except UnicodeDecodeError:
            gensim_obj = gensim.models.KeyedVectors.load_word2vec_format(self.embedding_file_path, binary=True)

        # Copying gensim object embedding matrix
        logging.debug("-> Fetching embedding matrix from model")
        self.embedding_matrix = gensim_obj.wv.syn0

        logging.debug("-> Matrix dimension: {}".format(self.embedding_matrix.shape))

        # Creating token-id mapping
        logging.debug("-> Creating word-id mapping")
        for i, item in enumerate(gensim_obj.wv.index2word, start=1):
            self.word_mapping[item] = i

        logging.debug("-> Creating padding vector (index=0)")
        pad_vector = np.random.rand(1, self.embedding_matrix.shape[1])
        self.embedding_matrix = np.insert(self.embedding_matrix, 0, pad_vector, axis=0)
        self.word_mapping["pad_token"] = 0

        # Deleting gensim object
        del gensim_obj
