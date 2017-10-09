import logging
import os

import numpy as np

from ..error import UnknownTokenAlreadyExists


class Embeddings:

    def __init__(self, embedding_file_path, embedding_oov_map_token_id):

        # Gensim model file path
        self.embedding_file_path = os.path.abspath(embedding_file_path)

        self.embedding_oov_map_token_id = embedding_oov_map_token_id

        # Word-id mapping
        self.word_mapping = dict()

        # Embedding matrix
        self.embedding_matrix = None

    def load_embedding(self):

        raise NotImplementedError

    def build_unknown_token(self):
        """
        Insert an 'unknown' token vector
        :return: nothing
        """

        if self.embedding_oov_map_token_id:
            raise Exception("The unknown token already exists")

        logging.debug("-> Creating random vector")
        unknown_vector = np.random.rand(1, self.embedding_matrix.shape[1])

        # Appending the unknown token vector to the embedding matrix
        logging.debug("-> Appending the unknown vector to the matrix and adding a mapping")
        self.embedding_matrix = np.append(self.embedding_matrix,
                                          unknown_vector,
                                          axis=0)

        # Creating a mapping for the unknown token vector
        logging.debug("-> Creating a mapping for the unknown token")
        self.embedding_oov_map_token_id = self._generate_unknown_token_id()

        logging.debug("-> Unknown vector ID: {}".format(self.embedding_oov_map_token_id))
        self.word_mapping[self.embedding_oov_map_token_id] = self.embedding_matrix.shape[0] - 1

    def _generate_unknown_token_id(self):

        tries = list()

        for i in range(1, 11):
            unknown_token_id = "{}UNK{}".format("#" * i, "#" * i)
            tries.append(unknown_token_id)
            if self.word_mapping.get(unknown_token_id):
                continue
            else:
                return unknown_token_id
        else:
            raise UnknownTokenAlreadyExists("Impossible to build an unknown token, all tested IDs already exist "
                                            "in the mapping: {}".format(" ".join(tries)))
