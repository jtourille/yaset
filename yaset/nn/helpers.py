import json
import os
import logging
import copy

from prettytable import PrettyTable


class TrainLogger:

    def __init__(self):

        self.iterations_log = dict()

    def __contains__(self, iteration_number):
        """
        Check if an iteration is present in the object
        :param iteration_number: iteration number
        :return: boolean
        """

        return iteration_number in self.iterations_log

    def store_minibatch_loss(self, ite, loss):

        self._create_iteration_item(ite)

        if "mini-batch-losses" not in self.iterations_log[ite]:
            self.iterations_log[ite]["mini-batch-losses"] = list()

        self.iterations_log[ite]["mini-batch-losses"].append(loss)

    def add_iteration_score(self, ite, score):
        """
        Add accuracy score for specific iteration
        :param ite: iteration number
        :param score: accuracy score
        :return: nothing
        """

        self._create_iteration_item(ite)

        self.iterations_log[ite]["dev_score"] = score

    def add_iteration_model_filename(self, ite, filename):
        """
        Add a model filename for a specific iteration
        :param ite: iteration number
        :param filename: model filename
        :return: nothing
        """

        self._create_iteration_item(ite)

        self.iterations_log[ite]["model_filename"] = os.path.basename(filename)

    def save_to_file(self, filename):
        """
        Save logger information to file (json format)
        :param filename: file where to dump information
        :return: nothing
        """

        payload = {
            "iterations": self.iterations_log
        }

        json.dump(payload, open(os.path.abspath(filename), "w", encoding="UTF-8"))

    def check_patience(self, patience):
        """
        Check if patience is reached for current training phase
        :param patience: patience parameter defined by user
        :return: boolean (True is patience is reached)
        """

        current_iteration = len(self.iterations_log) - 1

        score_list = [ite["dev_score"] for ite_nb, ite in sorted(self.iterations_log.items())]
        score_max = max(score_list)

        best_iteration = score_list.index(score_max)

        return current_iteration - best_iteration

    def get_score_table(self):
        """
        Return iteration performance on 'dev' part formatted as an ASCII table (prettytable object)
        :return: prettytable object
        """

        x = PrettyTable()
        best_ite = self.get_best_iteration()

        x.field_names = ["Iteration", "Dev Score"]

        for i, (iter_nb, payload) in enumerate(sorted(self.iterations_log.items()), start=1):

            if best_ite == i:
                current_iter_nb = "**{:03d}**".format(iter_nb)
            else:
                current_iter_nb = "{:03d}".format(iter_nb)
            current_score = "{:.5f}".format(payload["dev_score"])

            x.add_row([current_iter_nb, current_score])

        return x

    def get_best_iteration(self):

        score_list = [ite["dev_score"] for ite_nb, ite in sorted(self.iterations_log.items())]
        score_max = max(score_list)

        best_iteration = score_list.index(score_max) + 1

        return best_iteration

    def get_removable_iterations(self):
        """
        Get a list of least performing iteration for file removing (and disk space preserving)
        :return: list of iteration numbers
        """

        score_list = [ite["dev_score"] for ite_nb, ite in sorted(self.iterations_log.items())]
        score_max = max(score_list)

        best_iteration = score_list.index(score_max) + 1

        return [i for i in sorted(self.iterations_log) if i != best_iteration]

    def _create_iteration_item(self, ite):

        if ite not in self.iterations_log:
            self.iterations_log[ite] = dict()


def compute_bucket_boundaries(sequence_lengths, batch_size):
    """
    Compute bucket boundaries based on the sequence lengths
    :param sequence_lengths: sequence length to consider
    :param batch_size: mini-batch size used for learning
    :return: buckets boundaries (list)
    """

    nb_sequences = len(sequence_lengths)
    max_len = max(sequence_lengths)

    start = 0
    end = 10
    done = 0

    final_buckets = list()

    current_bucket = 0

    while done < nb_sequences:

        if nb_sequences - done < batch_size * 4:
            break

        for length in sequence_lengths:
            if start < length <= end:
                current_bucket += 1

        if current_bucket >= batch_size * 4:
            if start > 0:
                final_buckets.append(start)
            if end > 0:
                final_buckets.append(end)

            done += current_bucket
            start += 10
            end += 10

            current_bucket = 0
        else:
            end += 10

        final_buckets = sorted(list(set(final_buckets)))

    if len(final_buckets) >= 1:
        _ = final_buckets.pop(-1)

    if len(final_buckets) == 0:
        final_buckets.append(max_len+1)

    temp_buckets = copy.deepcopy(final_buckets)
    temp_buckets.append(0)
    temp_buckets.append(10000)

    count_total = 0

    for bigram in find_ngrams(sorted(temp_buckets), 2):
        count_current = 0

        for length in sequence_lengths:
            if bigram[0] < length <= bigram[1]:
                count_current += 1
                count_total += 1

        end = bigram[1]

        if end == 10000:
            logging.debug("* start={}+ | {:,} instances".format(bigram[0], count_current))
        else:
            logging.debug("* start={} -> end={} | {:,} instances".format(bigram[0], bigram[1], count_current))

    logging.debug("* TOTAL={:,}".format(count_total))

    return sorted(final_buckets)


def find_ngrams(input_list, n):

    return zip(*[input_list[i:] for i in range(n)])


def get_best_model(train_stats_file):
    """
    Return best training iteration file path
    :param train_stats_file: training log file path
    :return: filename
    """

    # Build model file path
    train_stats = json.load(open(os.path.abspath(train_stats_file), "r", encoding="UTF-8"))

    # Fetching iteration dev scores
    iterations = dict()

    for k, v in train_stats["iterations"].items():
        iterations[int(k)] = v

    score_list = [ite["dev_score"] for _, ite in sorted(iterations.items())]

    # Finding best score
    score_max = max(score_list)

    # Fetching iteration number
    best_iteration = score_list.index(score_max) + 1

    # Fetching filename
    best_filename = iterations[best_iteration]["model_filename"]

    return best_filename
