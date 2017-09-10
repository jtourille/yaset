import json
import logging
import os
import re
from collections import defaultdict

import tensorflow as tf
from sklearn.model_selection import train_test_split

from ..tools import ensure_dir


class TrainData:

    def __init__(self, train_data_file, dev_data_file=None, working_dir=None, dev_ratio=None):

        # Paths to tabulated data files
        self.train_data_file = train_data_file
        self.dev_data_file = dev_data_file

        # Paths to current working directory
        self.working_dir = working_dir

        # Percentage of documents to keep for development corpus in case there is no development data file
        # This number should be between 0 and 1
        self.dev_ratio = dev_ratio

        # Checking if train data file exists
        if not os.path.isfile(self.train_data_file):
            raise FileNotFoundError("The train file you specified doesn't exist: {}".format(self.train_data_file))

        # Checking if dev data file exists
        if self.dev_data_file and not os.path.isfile(self.dev_data_file):
            raise FileNotFoundError("The dev file you specified doesn't exist: {}".format(self.dev_data_file))

        # Path where TFRecords files will be stored
        self.tfrecords_dir_path = os.path.join(os.path.abspath(working_dir), "tfrecords")
        ensure_dir(self.tfrecords_dir_path)

        # Train and dev TFRecords file paths
        self.tfrecords_train_file = os.path.join(self.tfrecords_dir_path, "train.tfrecords")
        self.tfrecords_dev_file = os.path.join(self.tfrecords_dir_path, "dev.tfrecords")

        self.nb_train_instances = 0
        self.nb_dev_instances = 0

        self.length_train_instances = list()
        self.length_dev_instances = list()

        # UNKNOWN WORDS
        # ==================================
        self.nb_unknown_words_train = 0
        self.nb_unknown_words_dev = 0

        self.unknown_words_set_train = set()
        self.unknown_words_set_dev = set()

        self.unknown_word_file_train = os.path.join(working_dir, "unk_words_train.lst")
        self.unknown_word_file_dev = os.path.join(working_dir, "unk_words_dev.lst")
        # ==================================

        self.nb_words_train = 0
        self.nb_words_dev = 0

        self.label_mapping = list()

    def check_input_files(self):
        """
        Check input files (train and dev)
        :return: nothing
        """

        if self.train_data_file:
            logging.info("Checking train file")
            self._check_file(self.train_data_file)

        if self.dev_data_file:
            logging.info("Checking dev file")
            self._check_file(self.dev_data_file)

    def _check_file(self, data_file):
        """
        Check one input data file
        :param data_file: data file to check
        :return: nothing
        """

        labels = defaultdict(int)
        tokens = defaultdict(int)

        sequence_count = 0

        with open(data_file, "r", encoding="UTF-8") as input_file:

            current_sequence = 0

            for i, line in enumerate(input_file, start=1):

                if re.match("^$", line):
                    if current_sequence > 0:
                        current_sequence = 0
                        sequence_count += 1
                    continue

                parts = line.rstrip("\n").split("\t")
                current_sequence += 1

                if len(parts) < 2:
                    raise Exception("Error reading the input file at line {}: {}".format(i, data_file))

                tokens[parts[0]] += 1
                labels[parts[-1]] += 1

            if current_sequence > 0:
                sequence_count += 1

        logging.info("* Format: OK")
        logging.info("* nb. sequences: {:,}".format(sequence_count))
        logging.info("* nb. tokens: {:,}".format(sum([v for k, v in tokens.items()])))
        logging.info("* nb. labels: {:,}".format(len(labels)))
        for k, v in labels.items():
            logging.info("-> {}: {:,}".format(k, v))
            self.label_mapping.append(k)

    def create_tfrecords_files(self, embedding_object):
        """
        Create 'train' and 'dev' TFRecords files
        :param embedding_object: yaset embedding object to use for token IDs fetching
        :return: nothing
        """

        # Case where there is no dev data file
        if self.train_data_file and not self.dev_data_file:

            # Fetching number of sequences in the 'train' file
            sequence_nb_train = self._get_number_sequences(self.train_data_file)

            # Creating a list of sequence indexes based on the total number of sequences
            sequence_indexes = list(range(sequence_nb_train))

            # Dividing the index list into train and dev parts
            train_indexes, dev_indexes = train_test_split(sequence_indexes, test_size=self.dev_ratio, random_state=42)

            self.nb_train_instances = len(train_indexes)
            self.nb_dev_instances = len(dev_indexes)

            # Creating 'train' and 'dev' tfrecords files
            logging.info("Train...")
            self._convert_to_tfrecords(self.train_data_file, self.tfrecords_train_file,
                                       embedding_object, indexes=train_indexes, part="TRAIN")
            logging.info("* Nb. words: {:,}".format(self.nb_words_train))
            logging.info("* Nb. unknown words: {:,} ({:.2f}%)".format(
                self.nb_unknown_words_train,
                (self.nb_unknown_words_train / self.nb_words_train) * 100
            ))
            logging.info("* Nb. unique unknown words: {:,}".format(len(self.unknown_words_set_train)))
            logging.info("* Dumping unknown word list to file")
            self._dump_unknown_word_set(self.unknown_words_set_train, self.unknown_word_file_train)

            logging.info("Dev...")
            self._convert_to_tfrecords(self.train_data_file, self.tfrecords_dev_file,
                                       embedding_object, indexes=dev_indexes, part="DEV")

            logging.info("* Nb. words: {:,}".format(self.nb_words_dev))
            logging.info("* Nb. unknown words: {:,} ({:.2f}%)".format(
                self.nb_unknown_words_dev,
                (self.nb_unknown_words_dev / self.nb_words_dev) * 100
            ))
            logging.info("* Nb. unique unknown words: {:,}".format(len(self.unknown_words_set_dev)))
            logging.info("* Dumping unknown word list to file")
            self._dump_unknown_word_set(self.unknown_words_set_dev, self.unknown_word_file_dev)

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

    def _convert_to_tfrecords(self, data_file, target_tfrecords_file_path, embedding_object, indexes=None, part=None):
        """
        Create a TFRecords file
        :param data_file: source data files containing the sequences to write to the TFRecords file
        :param target_tfrecords_file_path: target TFRecords file path
        :param embedding_object: yaset embedding object used to fetch token IDs
        :param indexes: indexes of the sequences to write to the TFRecords file
        :return: nothing
        """

        # Will contains labels and tokens
        labels = list()
        tokens = list()

        sequence_id = 0

        writer = tf.python_io.TFRecordWriter(target_tfrecords_file_path)

        with open(data_file, "r", encoding="UTF-8") as input_file:

            current_sequence = 0

            for line in input_file:

                if re.match("^$", line):
                    if current_sequence > 0:
                        current_sequence = 0

                        if sequence_id in indexes:
                            self._write_example_to_file(writer, tokens, labels, embedding_object,
                                                        "{}-{}".format(part, sequence_id), part)

                        tokens.clear()
                        labels.clear()
                        sequence_id += 1

                    continue

                parts = line.rstrip("\n").split("\t")
                current_sequence += 1

                tokens.append(parts[0])
                labels.append(parts[-1])

            if current_sequence > 0:
                if sequence_id in indexes:
                    self._write_example_to_file(writer, tokens, labels, embedding_object,
                                                "{}-{}".format(part, sequence_id), part)

        writer.close()

    def _write_example_to_file(self, writer, tokens, labels, embedding_object, example_id, part):
        """
        Write an example to a TFRecords file
        :param writer: opened TFRecordWriter
        :param tokens: list of tokens
        :param labels: list of token labels
        :param embedding_object: yaset embedding object
        :return: nothing
        """

        if part == "TRAIN":
            self.nb_train_instances += 1
            self.length_train_instances.append(len(tokens))
        else:
            self.nb_dev_instances += 1
            self.length_dev_instances.append(len(tokens))

        example = tf.train.SequenceExample()

        example.context.feature["x_id"].bytes_list.value.append(
            tf.compat.as_bytes(example_id)
        )
        example.context.feature["x_length"].int64_list.value.append(len(tokens))

        x_tokens = example.feature_lists.feature_list["x_tokens"]
        y = example.feature_lists.feature_list["y"]

        for token, label in zip(tokens, labels):

            token_id = embedding_object.word_mapping.get(token)

            if part == "TRAIN":
                self.nb_words_train += 1
            else:
                self.nb_words_dev += 1

            if not token_id:
                token_id = embedding_object.word_mapping.get("##UNK##")
                if part == "TRAIN":
                    self.nb_unknown_words_train += 1
                    self.unknown_words_set_train.add(token)
                else:
                    self.nb_unknown_words_dev += 1
                    self.unknown_words_set_dev.add(token)

            label_id = self.label_mapping.index(label)

            x_tokens.feature.add().int64_list.value.append(token_id)
            y.feature.add().int64_list.value.append(label_id)

        writer.write(example.SerializeToString())

    @staticmethod
    def _dump_unknown_word_set(word_set, target_file):

        with open(target_file, "w", encoding="UTF-8") as output_file:
            for item in sorted(word_set):
                output_file.write("{}\n".format(item))

    def dump_data_characteristics(self, target_file, embedding_object):

        payload = {
            "label_mapping": self.label_mapping,
            "embedding_matrix_shape": embedding_object.embedding_matrix.shape
        }

        json.dump(payload, open(os.path.abspath(target_file), "w", encoding="UTF-8"))


class TestData:

    def __init__(self, test_data_file, working_dir=None, train_model_path=None):

        self.test_data_file = test_data_file
        self.working_dir = working_dir

        logging.info("Loading word mapping file")
        word_mapping_file = os.path.join(os.path.abspath(train_model_path), "word_mapping.json")
        self.word_mapping = json.load(open(word_mapping_file, "r", encoding="UTF-8"))

        logging.info("Loading data characteristics file")
        data_characteristics_file = os.path.join(os.path.abspath(train_model_path), "data_char.json")
        self.data_char = json.load(open(data_characteristics_file, "r", encoding="UTF-8"))

        self.label_mapping = self.data_char["label_mapping"]

        self.unknown_word_file = os.path.join(working_dir, "unk_words.lst")

        self.nb_instances = 0
        self.length_instances = list()
        self.nb_words = 0
        self.nb_unknown_words = 0
        self.unknown_words_set = set()

    def check_input_file(self):
        """
        Check input file
        :return: nothing
        """

        if self.test_data_file:
            logging.info("Checking file")
            self._check_file(self.test_data_file)

    @staticmethod
    def _check_file(data_file):
        """
        Check one input data file
        :param data_file: data file to check
        :return: nothing
        """

        tokens = defaultdict(int)
        columns = set()

        sequence_count = 0

        with open(data_file, "r", encoding="UTF-8") as input_file:

            current_sequence = 0

            for i, line in enumerate(input_file, start=1):

                if re.match("^$", line):
                    if current_sequence > 0:
                        current_sequence = 0
                        sequence_count += 1
                    continue

                parts = line.rstrip("\n").split("\t")
                current_sequence += 1
                columns.add(len(parts))

                if len(parts) < 1:
                    raise Exception("Error reading the input file at line {}: {}".format(i, data_file))

                tokens[parts[0]] += 1

            if current_sequence > 0:
                sequence_count += 1

        if len(columns) > 1:
            raise Exception("Error in the file, there is not the same number of columns")

        logging.info("* Format: OK")
        logging.info("* nb. sequences: {:,}".format(sequence_count))
        logging.info("* nb. tokens: {:,}".format(sum([v for k, v in tokens.items()])))

    def convert_to_tfrecords(self, data_file, target_tfrecords_file_path):
        """
        Create a TFRecords file
        :param data_file: source data files containing the sequences to write to the TFRecords file
        :param target_tfrecords_file_path: target TFRecords file path
        :return: nothing
        """

        logging.info("Creating TFRecords file")

        tokens = list()

        sequence_id = 0

        writer = tf.python_io.TFRecordWriter(target_tfrecords_file_path)

        with open(data_file, "r", encoding="UTF-8") as input_file:

            current_sequence = 0

            for line in input_file:

                if re.match("^$", line):
                    if current_sequence > 0:
                        current_sequence = 0

                        self._write_example_to_file(writer, tokens, "{}-{}".format("TEST", sequence_id))

                        tokens.clear()
                        sequence_id += 1

                    continue

                parts = line.rstrip("\n").split("\t")
                current_sequence += 1

                tokens.append(parts[0])

            if current_sequence > 0:
                self._write_example_to_file(writer, tokens, "{}-{}".format("TEST", sequence_id))

        writer.close()

        logging.info("* Nb. sequences: {:,}".format(sequence_id))
        logging.info("* Nb. words: {:,}".format(self.nb_words))
        logging.info("* Nb. unknown words: {:,} ({:.2f}%)".format(
            self.nb_unknown_words,
            (self.nb_unknown_words / self.nb_words) * 100
        ))
        logging.info("* Nb. unique unknown words: {:,}".format(len(self.unknown_words_set)))
        logging.info("Dumping unknown word list to file")
        self._dump_unknown_word_set(self.unknown_words_set, self.unknown_word_file)

    def _write_example_to_file(self, writer, tokens, example_id):
        """
        Write an example to a TFRecords file
        :param writer: opened TFRecordWriter
        :param tokens: list of tokens
        :param labels: list of token labels
        :param embedding_object: yaset embedding object
        :return: nothing
        """

        self.nb_instances += 1
        self.length_instances.append(len(tokens))

        example = tf.train.SequenceExample()

        example.context.feature["x_id"].bytes_list.value.append(
            tf.compat.as_bytes(example_id)
        )
        example.context.feature["x_length"].int64_list.value.append(len(tokens))

        x_tokens = example.feature_lists.feature_list["x_tokens"]

        for token in tokens:

            token_id = self.word_mapping.get(token)

            self.nb_words += 1

            if not token_id:
                token_id = self.word_mapping.get("##UNK##")
                self.nb_unknown_words += 1
                self.unknown_words_set.add(token)

            x_tokens.feature.add().int64_list.value.append(token_id)

        writer.write(example.SerializeToString())

    def write_predictions_to_file(self, target_file, pred_sequences):

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
                                    "\t".join(token + [self.label_mapping[pred]])
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
                            "\t".join(token + [self.label_mapping[pred]])
                        ))

    @staticmethod
    def _dump_unknown_word_set(word_set, target_file):

        with open(target_file, "w", encoding="UTF-8") as output_file:
            for item in sorted(word_set):
                output_file.write("{}\n".format(item))
