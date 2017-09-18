import json
import logging
import math
import os

import gensim
import numpy as np

from ..error import UnknownTokenAlreadyExists


class GensimEmbeddings:

    def __init__(self, embedding_file_path, embedding_unknown_token_id):

        # Gensim model file path
        self.embedding_file_path = os.path.abspath(embedding_file_path)

        # Word-id mapping
        self.word_mapping = dict()

        # Word count
        self.word_count = dict()

        # Embedding matrix
        self.embedding_matrix = None

        self.embedding_unknown_token_id = embedding_unknown_token_id

    def load_embedding(self):
        """
        Load embedding matrix and word count from gensim object
        :return: nothing
        """

        # Loading gensim object
        logging.debug("-> Loading gensim file")
        gensim_obj = gensim.models.Word2Vec.load(self.embedding_file_path)

        # Copying gensim object embedding matrix
        logging.debug("-> Fetching embedding matrix from gensim model")
        self.embedding_matrix = gensim_obj.wv.syn0

        # Creating token-id mapping
        logging.debug("-> Creating word-id mapping")
        for i, item in enumerate(gensim_obj.wv.index2word):
            self.word_mapping[item] = i

        # Fetching word count from gensim object
        logging.debug("-> Fetching word count from gensim model")
        for item in gensim_obj.wv.vocab:
            self.word_count[item] = gensim_obj.wv.vocab[item].count

        # Deleting gensim object
        del gensim_obj

    def build_unknown_token(self):
        """
        Build a unknown token vector for OOV token
        :return: nothing
        """

        # Raising exception if the unknown token already exists
        if self.word_mapping.get("##UNK##"):
            raise UnknownTokenAlreadyExists("The unknown token already exists")

        # Number of token to consider to compute the unknown token vector (1% of the vocabulary size)
        logging.debug("-> Computing number of vectors to use for averaging and creating empty matrix")
        nb_token_id_to_keep = math.ceil(len(self.word_count) * 0.01)

        # Creating an empty temporary matrix
        temp_matrix = np.empty((nb_token_id_to_keep, self.embedding_matrix.shape[1]))

        # List of 'n' least frequent tokens
        logging.debug("-> Fetching token ids")
        least_frequent_tokens = [k for k, f in sorted(self.word_count.items(), reverse=True)][:nb_token_id_to_keep]
        least_frequent_token_ids = [self.word_mapping[k] for k in least_frequent_tokens]

        logging.debug("-> Averaging over {} vectors".format(len(least_frequent_token_ids)))

        # Populating the matrix
        for i, k in enumerate(least_frequent_token_ids):
            temp_matrix[i] = self.embedding_matrix[k]

        # Computing the unknown token vector by averaging the matrix along axis 0
        unknown_vector = np.mean(temp_matrix, axis=0)

        # Appending the unknown token vector to the embedding matrix
        logging.debug("-> Appending the vector to the matrix and adding a mapping")
        self.embedding_matrix = np.append(self.embedding_matrix,
                                          unknown_vector.reshape(1, self.embedding_matrix.shape[1]),
                                          axis=0)

        # Creating a mapping for the unknown token vector
        self.word_mapping["##UNK##"] = self.embedding_matrix.shape[0] - 1
        self.embedding_unknown_token_id = "##UNK##"

        return "##UNK##"

    def dump_word_mapping(self, target_file):

        json.dump(self.word_mapping, open(target_file, "w", encoding="UTF-8"))
