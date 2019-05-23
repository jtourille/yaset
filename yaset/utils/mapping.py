import os
import re

import numpy as np
from gensim.models.word2vec import Word2Vec


def extract_ner_labels(instance_file: str = None):

    ner_labels = set()

    with open(os.path.abspath(instance_file), "r", encoding="UTF-8") as input_file:
        for line in input_file:
            if re.match("^$", line):
                continue

            parts = line.rstrip("\n").split("\t")
            ner_labels.add(parts[-1])

    ner_labels = sorted(list(ner_labels))

    return ner_labels


def extract_mappings_and_pretrained_matrix(options: dict = None,
                                           unk_symbol: str = "<unk>",
                                           pad_symbol: str = "<pad>") -> (dict, np.ndarray):

    all_mappings = dict()
    pretrained_matrix = None

    if options.get("embeddings").get("pretrained").get("use"):
        pretrained_configuration = options.get("embeddings").get("pretrained")

        if pretrained_configuration.get("format") == "gensim":
            model_filepath = os.path.abspath(pretrained_configuration.get("model_path"))
            embedding_obj = Word2Vec.load(model_filepath)

            all_mappings["tokens"] = [pad_symbol, unk_symbol] + embedding_obj.wv.index2word
            all_mappings["tokens"] = {v: k for k, v in enumerate(all_mappings["tokens"])}

            pretrained_matrix = embedding_obj.wv.syn0
            for _ in range(2):
                vector = np.random.rand(1, pretrained_matrix.shape[1])
                pretrained_matrix = np.append(pretrained_matrix, vector, axis=0)

        else:
            raise Exception("YASET supports only gensim embeddings")

    else:
        all_mappings["tokens"] = dict()

    ner_labels = extract_ner_labels(instance_file=options.get("data").get("train_file"))
    all_mappings["ner_labels"] = {v: k for k, v in enumerate(ner_labels)}

    all_mappings["characters"] = {
        "<bow>": 256,
        "<eow>": 257,
        "<pad>": 258
    }

    return all_mappings, pretrained_matrix
