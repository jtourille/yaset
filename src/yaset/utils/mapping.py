import logging
import os
import re

import numpy as np
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec


def extract_label_mapping(instance_file: str = None):

    ner_labels = set()

    with open(
        os.path.abspath(instance_file), "r", encoding="UTF-8"
    ) as input_file:
        for line in input_file:
            if re.match("^$", line) or re.match("^#", line):
                continue

            parts = line.rstrip("\n").split("\t")
            ner_labels.add(parts[-1])

    ner_labels = sorted(list(ner_labels))

    label_mapping = {v: k for k, v in enumerate(sorted(ner_labels))}

    return label_mapping


def extract_char_mapping(instance_file: str = None):

    char_literal = set()

    with open(
        os.path.abspath(instance_file), "r", encoding="UTF-8"
    ) as input_file:
        for line in input_file:
            if re.match("^$", line) or re.match("^#", line):
                continue

            parts = line.rstrip("\n").split("\t")
            for char in parts[0]:
                char_literal.add(char)

    char_literal.add("<bow>")
    char_literal.add("<eow>")
    char_literal.add("<pad>")

    char_literal = sorted(list(char_literal))

    char_mapping = {
        "char_literal": {k: i for i, k in enumerate(char_literal)},
        "char_utf8": {
            "<bow>": 256,
            "<eow>": 257,
            "<pad>": 258,
        },
    }

    return char_mapping


def extract_mappings_and_pretrained_matrix(
    options: dict = None,
    oov_symbol: str = "<unk>",
    pad_symbol: str = "<pad>",
    output_dir: str = None,
) -> (dict, np.ndarray):
    """
    Extract pretrained embedding matrix, size and mapping.

    :param output_dir: model output directory
    :type output_dir: str
    :param options: model parameters
    :type options: dict
    :param oov_symbol: symbol to use for OOV (vector will be created if necessary)
    :type oov_symbol: str
    :param pad_symbol: symbol to use for padding (vector will be created if necessary)
    :type pad_symbol: str

    :return: pretrained matrix, pretrained matrix size and pretrained matrix mapping
    :rtype: np.ndarray, int, dict
    """

    pretrained_matrix = None
    pretrained_matrix_size = None
    pretrained_matrix_mapping = None

    if options.get("embeddings").get("pretrained").get("use"):
        pretrained_matrix_mapping = dict()
        pretrained_configuration = options.get("embeddings").get("pretrained")

        updated_model_path = os.path.join(
            output_dir,
            "embeddings",
            "pretrained",
            os.path.basename(pretrained_configuration.get("model_path")),
        )

        if pretrained_configuration.get("format") == "gensim":
            embedding_obj = Word2Vec.load(updated_model_path)

        elif pretrained_configuration.get("format") == "glove":
            target_w2v_filepath = os.path.join(
                output_dir, "embeddings", "pretrained", "model.w2v"
            )

            logging.debug("Converting Glove embeddings to word2vec format")
            _ = glove2word2vec(updated_model_path, target_w2v_filepath)

            embedding_obj = KeyedVectors.load_word2vec_format(
                target_w2v_filepath
            )

        else:
            raise Exception(
                "Unsupported embedding type: {}".format(
                    pretrained_configuration.get("format")
                )
            )

        pretrained_matrix = embedding_obj.wv.syn0
        pretrained_matrix_index = embedding_obj.wv.index2word

        pretrained_matrix_mapping["tokens"] = list()

        if pad_symbol not in pretrained_matrix_index:
            pad_vector = np.random.rand(1, pretrained_matrix.shape[1])
            pretrained_matrix = np.append(
                pretrained_matrix, pad_vector, axis=0
            )
            pretrained_matrix_index.append(pad_symbol)

        if oov_symbol not in pretrained_matrix_index:
            unk_vector = np.random.rand(1, pretrained_matrix.shape[1])
            pretrained_matrix = np.append(
                pretrained_matrix, unk_vector, axis=0
            )
            pretrained_matrix_index.append(oov_symbol)

        pretrained_matrix_mapping["tokens"].extend(pretrained_matrix_index)
        pretrained_matrix_mapping["oov_id"] = pretrained_matrix_index.index(
            oov_symbol
        )
        pretrained_matrix_mapping["pad_id"] = pretrained_matrix_index.index(
            pad_symbol
        )

        pretrained_matrix_mapping["tokens"] = {
            v: k for k, v in enumerate(pretrained_matrix_mapping["tokens"])
        }

        pretrained_matrix_size = pretrained_matrix.shape

    return pretrained_matrix, pretrained_matrix_size, pretrained_matrix_mapping
