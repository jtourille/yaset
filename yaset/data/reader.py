import logging
import os
import re
from collections import defaultdict


class Data:

    def __init__(self, train_data_file, dev_data_file=None, test_data_file=None, gensim_model_path=None):

        self.train_data_file = train_data_file
        self.dev_data_file = dev_data_file
        self.test_data_file = test_data_file

        self.gensim_model_path = gensim_model_path

        if not os.path.isfile(self.train_data_file):
            raise FileNotFoundError("The train file you specified doesn't exist: {}".format(self.train_data_file))

        if self.dev_data_file and not os.path.isfile(self.dev_data_file):
            raise FileNotFoundError("The dev file you specified doesn't exist: {}".format(self.dev_data_file))

        if self.test_data_file and not os.path.isfile(self.test_data_file):
            raise FileNotFoundError("The dev file you specified doesn't exist: {}".format(self.test_data_file))

        if not os.path.isfile(self.gensim_model_path):
            raise FileNotFoundError("The gensim model you specified doesn't exist: {}".format(self.gensim_model_path))

        self._check_input_files()

    def _check_input_files(self):

        logging.info("CHECKING DATA FILES")
        logging.info("===================")

        if self.train_data_file:
            logging.info("** Checking train file")
            self._check_file(self.train_data_file)

        if self.dev_data_file:
            logging.info("** Checking dev file")
            self._check_file(self.dev_data_file)

        if self.test_data_file:
            logging.info("** Checking test file")
            self._check_file(self.test_data_file)

    @staticmethod
    def _check_file(data_file):

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

        logging.info("Format: OK")
        logging.info("nb. sequences: {}".format(sequence_count))
        logging.info("nb. tokens: {}".format(sum([v for k, v in tokens.items()])))
        logging.info("nb. labels: {}".format(len(labels)))
        for k, v in labels.items():
            logging.info("-> {}: {}".format(k, v))
