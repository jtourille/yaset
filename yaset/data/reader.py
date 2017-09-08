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

        self.nb_train_instances = None
        self.nb_dev_instances = None

        self.length_train_instances = None
        self.length_dev_instances = None

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
            self.length_train_instances = self._convert_to_tfrecords(self.train_data_file, self.tfrecords_train_file,
                                                                     embedding_object, indexes=train_indexes,
                                                                     part="TRAIN")

            logging.info("Dev...")
            self.length_dev_instances = self._convert_to_tfrecords(self.train_data_file, self.tfrecords_dev_file,
                                                                   embedding_object, indexes=dev_indexes,
                                                                   part="DEV")

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

        length_sequences = list()

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
                                                        "{}-{}".format(part, sequence_id))
                            length_sequences.append(len(tokens))

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
                                                "{}-{}".format(part, sequence_id))
                    length_sequences.append(len(tokens))

        writer.close()

        return length_sequences

    def _write_example_to_file(self, writer, tokens, labels, embedding_object, example_id):
        """
        Write an example to a TFRecords file
        :param writer: opened TFRecordWriter
        :param tokens: list of tokens
        :param labels: list of token labels
        :param embedding_object: yaset embedding object
        :return: nothing
        """

        example = tf.train.SequenceExample()

        example.context.feature["x_id"].bytes_list.value.append(
            tf.compat.as_bytes(example_id)
        )
        example.context.feature["x_length"].int64_list.value.append(len(tokens))

        x_tokens = example.feature_lists.feature_list["x_tokens"]
        y = example.feature_lists.feature_list["y"]

        for token, label in zip(tokens, labels):

            token_id = embedding_object.word_mapping.get(token)

            if not token_id:
                token_id = embedding_object.word_mapping.get("##UNK##")

            label_id = self.label_mapping.index(label)

            x_tokens.feature.add().int64_list.value.append(token_id)
            y.feature.add().int64_list.value.append(label_id)

        writer.write(example.SerializeToString())
