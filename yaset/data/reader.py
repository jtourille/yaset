import json
import logging
import os
import random
import re
from collections import defaultdict

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from ..error import FeatureDoesNotExist
from ..tools import ensure_dir


class StatsCorpus:

    def __init__(self, name):

        self.name = name
        self.nb_instances = 0
        self.sequence_lengths = list()
        self.nb_words = 0
        self.replaced_singletons = 0

        self.unknown_words = list()

    def log_stats(self):

        logging.info("* Nb. sequences: {:,}".format(len(self.sequence_lengths)))
        logging.info("* Nb. tokens: {:,}".format(self.nb_words))
        logging.info("* Nb. true unknown tokens: {:,} ({:.2f}%)".format(
            len(self.unknown_words),
            (len(self.unknown_words) / self.nb_words) * 100
        ))
        logging.info("* Nb. true unique unknown tokens: {:,}".format(len(list(set(self.unknown_words)))))
        logging.info("* Nb. replaced singletons: {:,}".format(self.replaced_singletons))

    def dump_unknown_tokens(self, target_file):

        with open(target_file, "w", encoding="UTF-8") as output_file:
            for item in sorted(list(set(self.unknown_words))):
                output_file.write("{}\n".format(item))


class TrainData:
    """
    Main class for training data
    """

    def __init__(self, working_dir=None, data_params=None):

        # Path to tabulated train files
        self.train_file_path = os.path.abspath(data_params.get("train_file_path"))

        # Checking if train file exists
        if not os.path.isfile(self.train_file_path):
            raise FileNotFoundError("The train file you specified does not exist: {}".format(self.train_file_path))

        # Extracting dev-instances-related parameters
        self.dev_file_use = data_params.get("dev_file_use")
        self.dev_random_seed_use = data_params.get("dev_random_seed_use")
        self.dev_random_seed_value = None
        self.dev_ratio = None

        if self.dev_file_use:
            self.dev_file_path = os.path.abspath(data_params.get("dev_file_path"))

            # Checking if dev file exists
            if not os.path.isfile(self.dev_file_path):
                raise FileNotFoundError("The 'dev' file you specified does not exist: {}".format(
                    self.dev_file_path
                ))
        else:
            self.dev_ratio = data_params.get("dev_random_ratio")

            if self.dev_ratio <= 0 or self.dev_ratio >= 1:
                raise Exception("The 'dev' ratio must be between 0 and 1 (current ratio: {})".format(self.dev_ratio))

            if self.dev_random_seed_use:
                self.dev_random_seed_value = data_params.get("dev_random_seed_value")

        # Paths to current working directory
        self.working_dir = working_dir

        self.lower_input = data_params.get("preproc_lower_input")
        self.replace_digits = data_params.get("replace_digits")

        self.singletons = None

        self.feature_use = data_params.get("feature_use")
        self.feature_columns = list()

        # -----------------------------------------------------------

        # Path where TFRecords files will be stored
        self.tfrecords_dir_path = os.path.join(os.path.abspath(working_dir), "tfrecords")

        # Train and dev TFRecords file paths
        self.tfrecords_train_file = os.path.join(self.tfrecords_dir_path, "train.tfrecords")
        self.tfrecords_dev_file = os.path.join(self.tfrecords_dir_path, "dev.tfrecords")

        # Train and dev unknown token lists
        self.unknown_tokens_train_file = os.path.join(self.working_dir, "unknown_tokens_train.lst")
        self.unknown_tokens_dev_file = os.path.join(self.working_dir, "unknown_tokens_dev.lst")

        # -----------------------------------------------------------

        self.train_stats = StatsCorpus(name="TRAIN")
        self.dev_stats = StatsCorpus(name="DEV")

        # -----------------------------------------------------------

        self.label_mapping = dict()
        self.inv_label_mapping = dict()

        self.char_mapping = dict()

        self.feature_value_mapping = dict()
        self.feature_nb = 0

    def check_input_files(self):
        """
        Check input file formats (train and dev if available)
        :return: nothing
        """

        if self.train_file_path:
            logging.info("Checking train file: {}".format(os.path.basename(self.train_file_path)))
            self._check_file(self.train_file_path, self.feature_columns)

        if self.dev_file_use:
            logging.info("Checking dev file: {}".format(os.path.basename(self.dev_file_path)))
            self._check_file(self.dev_file_path, self.feature_columns)

    def create_tfrecords_files(self, embedding_object, oov_strategy=None, unk_token_rate=None):
        """
        Create 'train' and 'dev' TFRecords files
        :param oov_strategy: Out-Of-Vocabulary strategy applied during training
        :param unk_token_rate: singleton replacement rate if applicable
        :param embedding_object: yaset embedding object to use for token IDs fetching
        :return: nothing
        """

        ensure_dir(self.tfrecords_dir_path)

        logging.debug("Lowercase: {}".format(self.lower_input))
        logging.debug("Replace digits: {}".format(self.replace_digits))
        if oov_strategy == "replace":
            logging.debug("OOV strategy: replace")
            logging.debug("Unknown token replacement rate: {}".format(unk_token_rate))

        elif oov_strategy == "map":
            logging.debug("OOV strategy: map")

        # Case where there is no dev data file
        if not self.dev_file_use:

            # Fetching number of sequences in the 'train' file
            sequence_nb_train = self._get_number_sequences(self.train_file_path)

            # Creating a list of sequence indexes based on the total number of sequences
            sequence_indexes = list(range(sequence_nb_train))

            # Dividing the index list into train and dev parts
            if self.dev_random_seed_use:
                train_indexes, dev_indexes = train_test_split(sequence_indexes, test_size=self.dev_ratio,
                                                              random_state=self.dev_random_seed_value)
            else:
                train_indexes, dev_indexes = train_test_split(sequence_indexes, test_size=self.dev_ratio)

            self._check_split(train_indexes, dev_indexes, self.train_file_path, self.train_file_path,
                              self.feature_columns)

            self.train_stats.nb_instances = len(train_indexes)
            self.dev_stats.nb_instances = len(dev_indexes)

            if oov_strategy == "replace":
                logging.info("Fetching singleton list")
                self.singletons = self._get_singletons(self.train_file_path, indexes=train_indexes,
                                                       lower_input=self.lower_input, replace_digits=self.replace_digits)
                logging.info("* Nb. singletons in train instances: {}".format(len(self.singletons)))

            logging.info("Building character mapping")
            self.char_mapping = self._get_char_mapping(self.train_file_path, train_indexes,
                                                       replace_digits=self.replace_digits)
            logging.info("* Nb. unique characters: {:,}".format(len(self.char_mapping) - 1))

            if self.feature_use:
                logging.info("Building attribute mapping")
                self.feature_value_mapping = self._get_feature_value_mapping(self.train_file_path, self.feature_columns,
                                                                             indexes=train_indexes)
                for i, col in enumerate(self.feature_columns, start=1):
                    logging.info("* Nb. unique values for feat. {} (col. #{}): {}".format(
                        i, col, len(self.feature_value_mapping[col])
                    ))
                    self.feature_nb += len(self.feature_value_mapping[col])

            logging.info("Building label mapping")
            self.label_mapping, self.inv_label_mapping = self._get_label_mapping(self.train_file_path, train_indexes)
            logging.info("* Nb. unique labels: {:,}".format(len(self.label_mapping)))

            # Creating 'train' and 'dev' tfrecords files
            logging.info("Creating TFRecords file for train instances...")

            self._convert_to_tfrecords(self.train_file_path, self.tfrecords_train_file,
                                       embedding_object, indexes=train_indexes, part="TRAIN",
                                       oov_strategy=oov_strategy, unk_token_rate=unk_token_rate)

            self.train_stats.log_stats()

            # Dumping unknown word set to working dir
            logging.info("* Dumping unknown word list to file")
            self.train_stats.dump_unknown_tokens(self.unknown_tokens_train_file)

            logging.info("Creating TFRecords file for dev instances...")
            self._convert_to_tfrecords(self.train_file_path, self.tfrecords_dev_file,
                                       embedding_object, indexes=dev_indexes, part="DEV",
                                       oov_strategy=oov_strategy, unk_token_rate=unk_token_rate)

            self.dev_stats.log_stats()

            # Dumping unknown word set to working dir
            logging.info("* Dumping unknown word list to file")
            self.dev_stats.dump_unknown_tokens(self.unknown_tokens_dev_file)

        else:

            sequence_nb_train = self._get_number_sequences(self.train_file_path)
            sequence_nb_dev = self._get_number_sequences(self.dev_file_path)

            train_indexes = list(range(sequence_nb_train))
            dev_indexes = list(range(sequence_nb_dev))

            self._check_split(train_indexes, dev_indexes, self.train_file_path, self.dev_file_path,
                              self.feature_columns)

            self.train_stats.nb_instances = len(train_indexes)
            self.dev_stats.nb_instances = len(dev_indexes)

            if oov_strategy == "replace":
                logging.info("Fetching singleton list")
                self.singletons = self._get_singletons(self.train_file_path, indexes=train_indexes,
                                                       lower_input=self.lower_input, replace_digits=self.replace_digits)
                logging.info("* Nb. singletons in train instances: {}".format(len(self.singletons)))

            logging.info("Building character mapping")
            self.char_mapping = self._get_char_mapping(self.train_file_path, train_indexes,
                                                       replace_digits=self.replace_digits)
            logging.info("* Nb. unique characters: {:,}".format(len(self.char_mapping) - 1))

            if self.feature_use:
                logging.info("Building attribute mapping")
                self.feature_value_mapping = self._get_feature_value_mapping(self.train_file_path, self.feature_columns,
                                                                             indexes=train_indexes)
                for i, col in enumerate(self.feature_columns, start=1):
                    logging.info("* Nb. unique values for feat. {} (col. #{}): {}".format(
                        i, col, len(self.feature_value_mapping[col])
                    ))
                    self.feature_nb += len(self.feature_value_mapping[col])

            logging.info("Building label mapping")
            self.label_mapping, self.inv_label_mapping = self._get_label_mapping(self.train_file_path, train_indexes)
            logging.info("* Nb. unique labels: {:,}".format(len(self.label_mapping)))

            # Creating 'train' and 'dev' tfrecords files
            logging.info("Creating TFRecords file for train instances...")

            self._convert_to_tfrecords(self.train_file_path, self.tfrecords_train_file,
                                       embedding_object, indexes=train_indexes, part="TRAIN",
                                       oov_strategy=oov_strategy, unk_token_rate=unk_token_rate)

            self.train_stats.log_stats()

            # Dumping unknown word set to working dir
            logging.info("* Dumping unknown word list to file")
            self.train_stats.dump_unknown_tokens(self.unknown_tokens_train_file)

            logging.info("Creating TFRecords file for dev instances...")
            self._convert_to_tfrecords(self.dev_file_path, self.tfrecords_dev_file,
                                       embedding_object, indexes=dev_indexes, part="DEV",
                                       oov_strategy=oov_strategy, unk_token_rate=unk_token_rate)

            self.dev_stats.log_stats()

            # Dumping unknown word set to working dir
            logging.info("* Dumping unknown word list to file")
            self.dev_stats.dump_unknown_tokens(self.unknown_tokens_dev_file)

    def _convert_to_tfrecords(self, data_file, target_tfrecords_file_path, embedding_object, indexes=None, part=None,
                              oov_strategy=None, unk_token_rate=None):
        """
        Create a TFRecords file
        :param data_file: source data files containing the sequences to write to the TFRecords file
        :param target_tfrecords_file_path: target TFRecords file path
        :param embedding_object: yaset embedding object used to fetch token IDs
        :param indexes: indexes of the sequences to write to the TFRecords file
        :return: nothing
        """

        # Will contains labels and tokens
        tokens = list()

        sequence_id = 0

        writer = tf.python_io.TFRecordWriter(target_tfrecords_file_path)

        with open(data_file, "r", encoding="UTF-8") as input_file:

            current_sequence = 0

            for line in input_file:

                if re.match("^$", line):
                    if current_sequence > 0:
                        current_sequence = 0

                        if indexes:
                            if sequence_id in indexes:
                                self._write_example_to_file(writer, tokens, embedding_object,
                                                            "{}-{}".format(part, sequence_id), part,
                                                            oov_strategy=oov_strategy,
                                                            unk_token_rate=unk_token_rate)
                        else:
                            self._write_example_to_file(writer, tokens, embedding_object,
                                                        "{}-{}".format(part, sequence_id), part,
                                                        oov_strategy=oov_strategy,
                                                        unk_token_rate=unk_token_rate)

                        tokens.clear()
                        sequence_id += 1

                    continue

                parts = line.rstrip("\n").split("\t")
                current_sequence += 1

                tokens.append(parts)

            if current_sequence > 0:
                if indexes:
                    if sequence_id in indexes:
                        self._write_example_to_file(writer, tokens, embedding_object,
                                                    "{}-{}".format(part, sequence_id), part,
                                                    oov_strategy=oov_strategy,
                                                    unk_token_rate=unk_token_rate)
                else:
                    self._write_example_to_file(writer, tokens, embedding_object,
                                                "{}-{}".format(part, sequence_id), part,
                                                oov_strategy=oov_strategy,
                                                unk_token_rate=unk_token_rate)

        writer.close()

    def _write_example_to_file(self, writer, tokens, embedding_object, example_id, part, oov_strategy=None,
                               unk_token_rate=None):
        """
        Write an example to a TFRecords file
        :param writer: opened TFRecordWriter
        :param tokens: list of tokens
        :param embedding_object: yaset embedding object
        :return: nothing
        """

        if part == "TRAIN":
            self.train_stats.sequence_lengths.append(len(tokens))
        else:
            self.dev_stats.sequence_lengths.append(len(tokens))

        example = tf.train.SequenceExample()

        example.context.feature["x_id"].bytes_list.value.append(
            tf.compat.as_bytes(example_id)
        )
        example.context.feature["x_length"].int64_list.value.append(len(tokens))

        x_tokens = example.feature_lists.feature_list["x_tokens"]
        x_chars = example.feature_lists.feature_list["x_chars"]
        x_chars_len = example.feature_lists.feature_list["x_chars_len"]
        y = example.feature_lists.feature_list["y"]

        x_atts = dict()
        for col in self.feature_columns:
            x_atts["x_att_{}".format(col)] = example.feature_lists.feature_list["x_att_{}".format(col)]

        token_max_size = 0

        for token in tokens:

            token_str = token[0]

            if self.lower_input:
                token_str = token_str.lower()

            if self.replace_digits:
                token_str = re.sub("\d", "0", token_str)

            token_id = embedding_object.word_mapping.get(token_str)

            if oov_strategy == "replace":
                if token_str in self.singletons and part == "TRAIN":
                    if random.random() < unk_token_rate:
                        token_id = embedding_object.word_mapping.get(embedding_object.embedding_oov_map_token_id)
                        self.train_stats.replaced_singletons += 1

            token_size = 0
            for char in token[0]:
                char_str = char

                if self.replace_digits:
                    char_str = re.sub("\d", "0", char_str)

                if char_str in self.char_mapping:
                    token_size += 1

            if token_size > token_max_size:
                token_max_size = token_size

            if part == "TRAIN":
                self.train_stats.nb_words += 1
            else:
                self.dev_stats.nb_words += 1

            if not token_id:
                token_id = embedding_object.word_mapping.get(embedding_object.embedding_oov_map_token_id)
                if part == "TRAIN":
                    self.train_stats.unknown_words.append(token[0])
                else:
                    self.dev_stats.unknown_words.append(token[0])

            label_id = self.label_mapping.get(token[-1])

            x_tokens.feature.add().int64_list.value.append(token_id)
            y.feature.add().int64_list.value.append(label_id)

            for col in self.feature_columns:
                feat_id = self.feature_value_mapping[col].get(token[col])
                x_atts["x_att_{}".format(col)].feature.add().int64_list.value.append(feat_id)

        for token in tokens:
            token_size = 0

            for char in token[0]:
                char_str = char
                if self.replace_digits:
                    char_str = re.sub("\d", "0", char_str)

                if char_str in self.char_mapping:
                    x_chars.feature.add().int64_list.value.append(self.char_mapping[char_str])
                    token_size += 1

            if token_size == 0:
                x_chars.feature.add().int64_list.value.append(0)
                token_size += 1

            x_chars_len.feature.add().int64_list.value.append(token_size)

            while token_size < token_max_size:
                x_chars.feature.add().int64_list.value.append(0)
                token_size += 1

        writer.write(example.SerializeToString())

    def dump_data_characteristics(self, target_file, embedding_object):

        payload = {
            "label_mapping": self.label_mapping,
            "embedding_matrix_shape": embedding_object.embedding_matrix.shape,
            "word_mapping": embedding_object.word_mapping,
            "char_mapping": self.char_mapping,
            "feature_value_mapping": self.feature_value_mapping,
            "feature_nb": self.feature_nb,
            "feature_columns": self.feature_columns,
            "embedding_unknown_token_id": embedding_object.embedding_unknown_token_id,
            "lower_input": self.lower_input,
            "replace_digits": self.replace_digits
        }

        json.dump(payload, open(os.path.abspath(target_file), "w", encoding="UTF-8"))

    def _check_split(self, train_indexes, dev_indexes, train_data_file, dev_data_file, features_columns):
        """
        Check if all attributes values from dev instances will be seen in the train part
        :param train_indexes: train instance indexes
        :param dev_indexes: dev instance indexes
        :param train_data_file: file from which train instances are extracted
        :param dev_data_file: file from which dev instances are extracted
        :param features_columns: feature column indexes
        :return: nothing
        """

        # Fetching labels and attribute values from train instances
        labels_train, attributes_train = self._get_attributes_and_labels(train_data_file, train_indexes,
                                                                         features_columns)

        # Fetching labels and attribute values from dev instances
        labels_dev, attributes_dev = self._get_attributes_and_labels(dev_data_file, dev_indexes, features_columns)

        # Checking if attributes values from dev instances are present in train instances
        for col, att_dict in attributes_dev.items():
            for k, v in att_dict.items():
                if k not in attributes_train[col]:
                    logging.info("One feature value from col. #{} in dev corpus is not present in train "
                                 "corpus: {}".format(col, k))
                    if self.dev_file_path:
                        logging.info("Check your input files and relaunch yaset")
                    else:
                        logging.info("Try to relaunch yaset after changing the random seed")

                    raise FeatureDoesNotExist("A feature value at col. #{} from dev instances is not present in train"
                                              " instances: {}".format(col, k))

    @staticmethod
    def _get_singletons(data_file, indexes=None, lower_input=None, replace_digits=None):
        """
        Compute the character-mapping file
        :param data_file: source data files containing the sequences to write to the TFRecords file
        :param indexes: indexes of the sequences to write to the TFRecords file
        :return: nothing
        """

        # Will contains labels and tokens
        tokens_count = defaultdict(int)
        singletons = set()
        tokens = list()

        sequence_id = 0

        with open(data_file, "r", encoding="UTF-8") as input_file:

            current_tokens = list()

            for line in input_file:

                if re.match("^$", line):
                    if len(current_tokens) > 0:
                        if indexes:
                            if sequence_id in indexes:
                                tokens = tokens + current_tokens
                        else:
                            tokens = tokens + current_tokens

                        current_tokens.clear()
                        sequence_id += 1

                    continue

                parts = line.rstrip("\n").split("\t")

                tokens.append(parts[0])

            if len(current_tokens) > 0:
                if indexes:
                    if sequence_id in indexes:
                        tokens = tokens + current_tokens
                else:
                    tokens = tokens + current_tokens

        for token in tokens:

            token_str = token

            if replace_digits:
                token_str = re.sub("\d", "0", token_str)

            if lower_input:
                token_str = token_str.lower()

            tokens_count[token_str] += 1

        for k, v in tokens_count.items():
            if v == 1:
                singletons.add(k)

        return singletons

    @staticmethod
    def _get_char_mapping(data_file, indexes=None, replace_digits=None):
        """
        Compute the character-mapping file
        :param data_file: source data files containing the sequences to write to the TFRecords file
        :param indexes: indexes of the sequences to write to the TFRecords file
        :return: nothing
        """

        # Will contains labels and tokens
        tokens = list()
        char_mapping = dict()

        sequence_id = 0

        with open(data_file, "r", encoding="UTF-8") as input_file:

            current_tokens = list()

            for line in input_file:

                if re.match("^$", line):
                    if len(current_tokens) > 0:
                        if indexes:
                            if sequence_id in indexes:
                                tokens = tokens + current_tokens
                        else:
                            tokens = tokens + current_tokens

                        current_tokens.clear()
                        sequence_id += 1

                    continue

                parts = line.rstrip("\n").split("\t")

                tokens.append(parts[0])

            if len(current_tokens) > 0:
                if indexes:
                    if sequence_id in indexes:
                        tokens = tokens + current_tokens
                else:
                    tokens = tokens + current_tokens

        char_set = set()
        for token in tokens:
            for char in token:
                char_str = char

                if replace_digits:
                    char_str = re.sub("\d", "0", char_str)

                char_set.add(char_str)

        for i, char in enumerate(sorted(char_set), start=1):
            char_mapping[char] = i

        char_mapping["pad_character"] = 0

        return char_mapping

    @staticmethod
    def _check_file(data_file, features_columns):
        """
        Check input data file format
        :param data_file: data file to check
        :return: nothing
        """

        labels = defaultdict(int)
        tokens = defaultdict(int)

        attributes = dict()
        for col in features_columns:
            attributes[col] = defaultdict(int)

        sequence_lengths = list()
        column_nb = set()

        sequence_count = 0

        with open(data_file, "r", encoding="UTF-8") as input_file:

            current_sequence = 0

            for i, line in enumerate(input_file, start=1):
                if re.match("^$", line):
                    if current_sequence > 0:
                        sequence_lengths.append(current_sequence)  # Appending current sequence length to list
                        current_sequence = 0  # Resetting length counter
                        sequence_count += 1  # Incrementing sequence length

                    continue

                parts = line.rstrip("\n").split("\t")  # Splitting line

                column_nb.add(len(parts))  # Keeping track of the number of columns
                current_sequence += 1

                # Raising exception if all lines do not have the same number of columns or
                # if the number of columns is < 2
                if len(column_nb) > 1 or len(parts) < 2:
                    raise Exception("Error reading the input file at line {}: {}".format(i, data_file))

                # Counting tokens and labels
                tokens[parts[0]] += 1
                labels[parts[-1]] += 1

                for col, val_dict in attributes.items():
                    val_dict[parts[col]] += 1

            # End of file, adding information about the last sequence if necessary
            if current_sequence > 0:
                sequence_count += 1
                sequence_lengths.append(current_sequence)

        logging.info("* format: OK")
        logging.info("* nb. sequences: {:,}".format(sequence_count))
        logging.info("* average sequence length: {:,.3f} (min={:,} max={:,} std={:,.3f})".format(
            np.mean(sequence_lengths),
            np.min(sequence_lengths),
            np.max(sequence_lengths),
            np.std(sequence_lengths)
        ))
        logging.info("* nb. tokens (col. #0): {:,} (unique={:,})".format(
            sum([v for k, v in tokens.items()]),
            len(tokens)
        ))
        for i, (col, val_dict) in enumerate(attributes.items(), start=1):
            nb_attributes = sum([v for k, v in val_dict.items()])
            logging.info("* nb. att. {} (col. #{}): {:,}".format(
                i,
                col,
                len(val_dict)
            ))
            for k, v in val_dict.items():
                logging.debug("-> {}: {:,} ({:.3f}%)".format(k, v, (v / nb_attributes) * 100))

        logging.info("* nb. labels (col. #{}): {:,}".format(list(column_nb)[0] - 1, len(labels)))
        nb_labels = sum([v for k, v in labels.items()])
        for k, v in labels.items():
            logging.debug("-> {}: {:,} ({:.3f}%)".format(k, v, (v/nb_labels) * 100))

    @staticmethod
    def _get_feature_value_mapping(data_file, feature_columns, indexes=None):

        # Will contains labels and tokens
        tokens = list()
        feature_value_mapping = dict()
        for col in feature_columns:
            feature_value_mapping[col] = dict()

        sequence_id = 0

        with open(data_file, "r", encoding="UTF-8") as input_file:

            current_tokens = list()

            for line in input_file:

                if re.match("^$", line):
                    if len(current_tokens) > 0:
                        if indexes:
                            if sequence_id in indexes:
                                tokens = tokens + current_tokens
                        else:
                            tokens = tokens + current_tokens

                        current_tokens.clear()
                        sequence_id += 1

                    continue

                parts = line.rstrip("\n").split("\t")

                tokens.append(parts)

            if len(current_tokens) > 0:
                if indexes:
                    if sequence_id in indexes:
                        tokens = tokens + current_tokens
                else:
                    tokens = tokens + current_tokens

        feature_index = 0

        for token in tokens:
            for col in feature_columns:
                if token[col] not in feature_value_mapping[col]:
                    feature_value_mapping[col][token[col]] = feature_index
                    feature_index += 1

        return feature_value_mapping

    @staticmethod
    def _get_label_mapping(data_file, indexes=None):
        """
        Compute the character-mapping file
        :param data_file: source data files containing the sequences to write to the TFRecords file
        :param indexes: indexes of the sequences to write to the TFRecords file
        :return: nothing
        """

        # Will contains labels and tokens
        labels = list()
        label_mapping = dict()
        inv_label_mapping = dict()

        sequence_id = 0

        with open(data_file, "r", encoding="UTF-8") as input_file:

            current_labels = list()

            for line in input_file:

                if re.match("^$", line):
                    if len(current_labels) > 0:
                        if indexes:
                            if sequence_id in indexes:
                                labels = labels + current_labels
                        else:
                            labels = labels + current_labels

                        current_labels.clear()
                        sequence_id += 1

                    continue

                parts = line.rstrip("\n").split("\t")

                current_labels.append(parts[-1])

            if len(current_labels) > 0:
                if indexes:
                    if sequence_id in indexes:
                        labels = labels + current_labels
                else:
                    labels = labels + current_labels

        label_set = set()
        for label in labels:
            label_set.add(label)

        for i, label in enumerate(sorted(label_set)):
            label_mapping[label] = i
            inv_label_mapping[i] = label

        return label_mapping, inv_label_mapping

    @staticmethod
    def _get_number_sequences(data_file_path):
        """
        Get the total number of sequences in a data file
        :param data_file_path: data file path
        :return: number of sequences
        """

        sequence_count = 0

        with open(data_file_path, "r", encoding="UTF-8") as input_file:

            current_sequence = 0

            for i, line in enumerate(input_file, start=1):

                if re.match("^$", line):
                    if current_sequence > 0:
                        current_sequence = 0
                        sequence_count += 1
                    continue

                current_sequence += 1

            if current_sequence > 0:
                sequence_count += 1

        return sequence_count

    @staticmethod
    def _get_attributes_and_labels(data_file, indexes, features_columns):
        """
        Get attribute and labels values from a group of instances
        :param data_file: file from which instances are extracted
        :param indexes: instance indexes
        :param features_columns: feature column indexes
        :return: labels and attributes
        """

        # Initializing variables
        labels = defaultdict(int)
        attributes = dict()
        for col in features_columns:
            attributes[col] = defaultdict(int)

        # Keeping track of sentence id
        sequence_id = 0
        current_sequence = list()

        with open(data_file, "r", encoding="UTF-8") as input_file:
            for line in input_file:
                if re.match("^$", line):
                    if len(current_sequence) > 0:
                        if sequence_id in indexes:
                            for tok in current_sequence:
                                labels[tok[-1]] += 1

                                for col in features_columns:
                                    attributes[col][tok[col]] += 1

                        sequence_id += 1
                        current_sequence.clear()

                    continue

                parts = line.rstrip("\n").split("\t")
                current_sequence.append(parts)

            if len(current_sequence) > 0:
                if sequence_id in indexes:
                    for tok in current_sequence:
                        labels[tok[-1]] += 1

                        for col in features_columns:
                            attributes[col][tok[col]] += 1

        return labels, attributes


class TestData:

    def __init__(self, test_data_file, working_dir=None, train_model_path=None):

        self.test_data_file = test_data_file
        self.working_dir = working_dir

        logging.info("Loading data characteristics file")
        data_characteristics_file = os.path.join(os.path.abspath(train_model_path), "data_char.json")
        self.data_char = json.load(open(data_characteristics_file, "r", encoding="UTF-8"))

        self.word_mapping = self.data_char["word_mapping"]
        self.embedding_unknown_token_id = self.data_char["embedding_unknown_token_id"]

        self.char_mapping = self.data_char["char_mapping"]
        self.feature_value_mapping = dict()
        for k, v in self.data_char["feature_value_mapping"].items():
            self.feature_value_mapping[int(k)] = v

        self.feature_nb = self.data_char["feature_nb"]
        self.feature_columns = self.data_char["feature_columns"]

        # Fetching label mapping from training data characteristics
        self.label_mapping = self.data_char["label_mapping"]

        self.lower_input = self.data_char["lower_input"]
        self.replace_digits = self.data_char["replace_digits"]

        self.test_stats = StatsCorpus(name="TEST")

    def check_input_file(self):
        """
        Check input file
        :return: nothing
        """

        if self.test_data_file:
            logging.info("Checking file")
            self._check_file(self.test_data_file, self.feature_columns)

    @staticmethod
    def _check_file(data_file, feature_columns):
        """
        Check input data file format
        :param data_file: data file to check
        :return: nothing
        """

        labels = defaultdict(int)
        tokens = defaultdict(int)

        attributes = dict()
        for col in feature_columns:
            attributes[col] = defaultdict(int)

        sequence_lengths = list()
        column_nb = set()

        sequence_count = 0

        with open(data_file, "r", encoding="UTF-8") as input_file:

            current_sequence = 0

            for i, line in enumerate(input_file, start=1):
                if re.match("^$", line):
                    if current_sequence > 0:
                        sequence_lengths.append(current_sequence)  # Appending current sequence length to list
                        current_sequence = 0  # Resetting length counter
                        sequence_count += 1  # Incrementing sequence length

                    continue

                parts = line.rstrip("\n").split("\t")  # Splitting line

                column_nb.add(len(parts))  # Keeping track of the number of columns
                current_sequence += 1

                # Raising exception if all lines do not have the same number of columns or
                # if the number of columns is < 2
                if len(column_nb) > 1 or len(parts) < 2:
                    raise Exception("Error reading the input file at line {}: {}".format(i, data_file))

                # Counting tokens and labels
                tokens[parts[0]] += 1
                labels[parts[-1]] += 1

                for col, val_dict in attributes.items():
                    val_dict[parts[col]] += 1

            # End of file, adding information about the last sequence if necessary
            if current_sequence > 0:
                sequence_count += 1
                sequence_lengths.append(current_sequence)

        logging.info("* format: OK")
        logging.info("* nb. sequences: {:,}".format(sequence_count))
        logging.info("* average sequence length: {:,.3f} (min={:,} max={:,} std={:,.3f})".format(
            np.mean(sequence_lengths),
            np.min(sequence_lengths),
            np.max(sequence_lengths),
            np.std(sequence_lengths)
        ))
        logging.info("* nb. tokens (col. #0): {:,} (unique={:,})".format(
            sum([v for k, v in tokens.items()]),
            len(tokens)
        ))
        for i, (col, val_dict) in enumerate(attributes.items(), start=1):
            nb_attributes = sum([v for k, v in val_dict.items()])
            logging.info("* nb. att. {} (col. #{}): {:,}".format(
                i,
                col,
                len(val_dict)
            ))
            for k, v in val_dict.items():
                logging.debug("-> {}: {:,} ({:.3f}%)".format(k, v, (v / nb_attributes) * 100))

        logging.info("* nb. labels (col. #{}): {:,}".format(list(column_nb)[0] - 1, len(labels)))
        nb_labels = sum([v for k, v in labels.items()])
        for k, v in labels.items():
            logging.debug("-> {}: {:,} ({:.3f}%)".format(k, v, (v/nb_labels) * 100))

    def convert_to_tfrecords(self, data_file, target_tfrecords_file_path):
        """
        Create a TFRecords file
        :param data_file: source data files containing the sequences to write to the TFRecords file
        :param target_tfrecords_file_path: target TFRecords file path
        :return: nothing
        """

        logging.info("Creating TFRecords file for test instances...")

        logging.debug("Lowercase: {}".format(self.lower_input))
        logging.debug("Replace digits: {}".format(self.replace_digits))

        sequence_nb = self._get_number_sequences(data_file)
        self._check_data(data_file, self.feature_columns, list(range(sequence_nb)))

        self.test_stats.nb_instances = sequence_nb

        tokens = list()

        sequence_id = 0

        # TFRecord writer
        writer = tf.python_io.TFRecordWriter(target_tfrecords_file_path)

        with open(data_file, "r", encoding="UTF-8") as input_file:

            for line in input_file:

                if re.match("^$", line):
                    if len(tokens) > 0:
                        self._write_example_to_file(writer, tokens, "{}-{}".format("TEST", sequence_id))

                        sequence_id += 1
                        tokens.clear()

                    continue

                parts = line.rstrip("\n").split("\t")
                tokens.append(parts)

            # End of file, dumping current sequence
            if len(tokens) > 0:
                self._write_example_to_file(writer, tokens, "{}-{}".format("TEST", sequence_id))
                sequence_id += 1

        self.test_stats.log_stats()

        writer.close()

    def _write_example_to_file(self, writer, tokens, example_id):
        """
        Write an example to a TFRecords file
        :param writer: opened TFRecordWriter
        :param tokens: list of tokens
        :return: nothing
        """

        self.test_stats.sequence_lengths.append(len(tokens))

        example = tf.train.SequenceExample()

        example.context.feature["x_id"].bytes_list.value.append(
            tf.compat.as_bytes(example_id)
        )
        example.context.feature["x_length"].int64_list.value.append(len(tokens))

        x_tokens = example.feature_lists.feature_list["x_tokens"]
        x_chars = example.feature_lists.feature_list["x_chars"]
        x_chars_len = example.feature_lists.feature_list["x_chars_len"]

        x_atts = dict()
        for col in self.feature_columns:
            x_atts["x_att_{}".format(col)] = example.feature_lists.feature_list["x_att_{}".format(col)]

        token_max_size = 0

        for token in tokens:

            token_str = token[0]

            if self.replace_digits:
                token_str = re.sub("\d", "0", token_str)

            if self.lower_input:
                token_str = token_str.lower()

            token_id = self.word_mapping.get(token_str)

            token_size = 0
            for char in token[0]:
                char_str = char

                if self.replace_digits:
                    char_str = re.sub("\d", "0", char_str)

                if char_str in self.char_mapping:
                    token_size += 1

            if token_size > token_max_size:
                token_max_size = token_size

            self.test_stats.nb_words += 1

            if not token_id:
                token_id = self.word_mapping.get(self.embedding_unknown_token_id)
                self.test_stats.unknown_words.append(token[0])

            x_tokens.feature.add().int64_list.value.append(token_id)

            for col in self.feature_columns:
                feat_id = self.feature_value_mapping[col].get(token[col])
                x_atts["x_att_{}".format(col)].feature.add().int64_list.value.append(feat_id)

        for token in tokens:
            token_size = 0

            for char in token[0]:
                char_str = char

                if self.replace_digits:
                    char_str = re.sub("\d", "0", char_str)

                if char_str in self.char_mapping:
                    x_chars.feature.add().int64_list.value.append(self.char_mapping[char_str])
                    token_size += 1

            if token_size == 0:
                x_chars.feature.add().int64_list.value.append(0)
                token_size += 1

            x_chars_len.feature.add().int64_list.value.append(token_size)

            while token_size < token_max_size:
                x_chars.feature.add().int64_list.value.append(0)
                token_size += 1

        writer.write(example.SerializeToString())

    def write_predictions_to_file(self, target_file, pred_sequences):

        inverse_label_mapping = dict()

        for k, v in self.label_mapping.items():
            inverse_label_mapping[v] = k

        with open(self.test_data_file, "r", encoding="UTF-8") as input_file:
            with open(os.path.abspath(target_file), "w", encoding="UTF-8") as output_file:

                sequence_parts = list()
                sequence_id = 0

                for line in input_file:
                    if re.match("^$", line):
                        if len(sequence_parts) > 0:
                            cur_pred_seq = pred_sequences["TEST-{}".format(sequence_id)]

                            for token, pred in zip(sequence_parts, cur_pred_seq):

                                output_file.write('{}\n'.format(
                                    "\t".join(token + [inverse_label_mapping[pred]])
                                ))

                            sequence_id += 1
                            sequence_parts.clear()

                        output_file.write("\n")
                        continue

                    parts = line.rstrip("\n").split("\t")
                    sequence_parts.append(parts)

                if len(sequence_parts) > 0:
                    cur_pred_seq = pred_sequences["TEST-{}".format(sequence_id)]

                    for token, pred in zip(sequence_parts, cur_pred_seq):
                        output_file.write('{}\n'.format(
                            "\t".join(token + [inverse_label_mapping[pred]])
                        ))

    @staticmethod
    def _dump_unknown_word_set(word_set, target_file):

        with open(target_file, "w", encoding="UTF-8") as output_file:
            for item in sorted(word_set):
                output_file.write("{}\n".format(item))

    def _check_data(self, data_file, features_columns, indexes):

        # Fetching labels and attribute values from train instances
        labels, attributes = self._get_attributes_and_labels(data_file, indexes, features_columns)

        # Checking if attributes values from dev instances are present in train instances
        for col, att_dict in attributes.items():
            for k, v in att_dict.items():
                if k not in self.feature_value_mapping[col]:
                    logging.info("One feature value from col. #{} in test instances was not present during "
                                 "training {}:".format(col, k))
                    logging.info("Check your input files and relaunch yaset")

                    raise FeatureDoesNotExist("A feature value at col. #{} from test instances was not seen during "
                                              "training: {}".format(col, k))

    @staticmethod
    def _get_attributes_and_labels(data_file, indexes, features_columns):
        """
        Get attribute and labels values from a group of instances
        :param data_file: file from which instances are extracted
        :param indexes: instance indexes
        :param features_columns: feature column indexes
        :return: labels and attributes
        """

        # Initializing variables
        labels = defaultdict(int)
        attributes = dict()
        for col in features_columns:
            attributes[col] = defaultdict(int)

        # Keeping track of sentence id
        sequence_id = 0
        current_sequence = list()

        with open(data_file, "r", encoding="UTF-8") as input_file:
            for line in input_file:
                if re.match("^$", line):
                    if len(current_sequence) > 0:
                        if sequence_id in indexes:
                            for tok in current_sequence:
                                labels[tok[-1]] += 1

                                for col in features_columns:
                                    attributes[col][tok[col]] += 1

                        sequence_id += 1
                        current_sequence.clear()

                    continue

                parts = line.rstrip("\n").split("\t")
                current_sequence.append(parts)

            if len(current_sequence) > 0:
                if sequence_id in indexes:
                    for tok in current_sequence:
                        labels[tok[-1]] += 1

                        for col in features_columns:
                            attributes[col][tok[col]] += 1

        return labels, attributes

    @staticmethod
    def _get_number_sequences(data_file_path):
        """
        Get the total number of sequences in a data file
        :param data_file_path: data file path
        :return: number of sequences
        """

        sequence_count = 0

        with open(data_file_path, "r", encoding="UTF-8") as input_file:

            current_sequence = 0

            for i, line in enumerate(input_file, start=1):

                if re.match("^$", line):
                    if current_sequence > 0:
                        current_sequence = 0
                        sequence_count += 1
                    continue

                current_sequence += 1

            if current_sequence > 0:
                sequence_count += 1

        return sequence_count
