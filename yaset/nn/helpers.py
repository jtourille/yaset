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

        for iter_nb, payload in sorted(self.iterations_log.items()):
            x.field_names = ["Ite. nb.", "dev acc."]

            current_iter_nb = "{:3d}".format(iter_nb)
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


def conll_evaluation(sequences):
    """
    Perform CoNLL-like evaluation. Implemented in python for ease of use.
    :param sequences: sequences of tokens. Each token is a dictionary with two attributes 'gs' and 'pred'
    :return: precision, recall and f1
    """

    # Fetching gold standard and predicted entities
    gs_entity = get_entities(sequences, "gs")
    pred_entity = get_entities(sequences, "pred")

    # Initializing counters for metric computation
    pred = 0
    corr = 0
    gs = 0

    # Counting correctly predicted entities
    for entity in gs_entity:
        if entity in pred_entity:
            corr += 1

    pred += len(pred_entity)
    gs += len(gs_entity)

    # Computing precision, recall and metrics
    try:
        precision = float(corr) / pred
        recall = float(corr) / gs
        f1 = (2 * precision * recall) / (precision + recall)

    except ZeroDivisionError:
        precision = 0.0
        recall = 0.0
        f1 = 0.0

    return precision, recall, f1


def get_entities(sequences, entity_type):
    """
    Get entities from sequences (predicted or gold-standard)
    :param sequences: list of sequences
    :param entity_type: entity type to fetch ("gs" or "pred")
    :return: list of entities
    """

    entities = list()

    for i, sequence in enumerate(sequences):

        previous_label = "O"
        current_cat = None
        current_tokens = list()

        for j, tok in enumerate(sequence):
            if tok[entity_type].startswith("B") or tok[entity_type].startswith("S"):

                if previous_label in ["I", "E"]:

                    # Clearing entity
                    new_entity = "{:d}_{}_{}".format(i, current_cat,
                                                     "-".join([str(item) for item in sorted(current_tokens)]))
                    entities.append(new_entity)
                    current_cat = None
                    current_tokens.clear()

                    # Starting new entity
                    current_cat = tok[entity_type].split("-")[1]
                    previous_label = "B"
                    current_tokens.append(j)

                elif previous_label in ["B", "S"]:

                    # Clearing entity
                    new_entity = "{:d}_{}_{}".format(i, current_cat,
                                                     "-".join([str(item) for item in sorted(current_tokens)]))
                    entities.append(new_entity)
                    current_cat = None
                    current_tokens.clear()

                    # Starting new entity
                    current_cat = tok[entity_type].split("-")[1]
                    previous_label = "B"
                    current_tokens.append(j)

                elif previous_label == "O":

                    # Starting new entity
                    current_cat = tok[entity_type].split("-")[1]
                    previous_label = "B"
                    current_tokens.append(j)

            elif tok[entity_type].startswith("O"):

                if previous_label in ["B", "I", "S", "E"]:

                    # Clearing entity
                    new_entity = "{:d}_{}_{}".format(i, current_cat,
                                                     "-".join([str(item) for item in sorted(current_tokens)]))
                    entities.append(new_entity)
                    current_cat = None
                    current_tokens.clear()
                    previous_label = "O"

            elif tok[entity_type].startswith("I") or tok[entity_type].startswith("E"):

                if previous_label == "O":

                    # Starting new entity
                    current_cat = tok[entity_type].split("-")[1]
                    previous_label = "B"
                    current_tokens.append(j)

                elif previous_label in ["B", "S"]:

                    token_cat = tok[entity_type].split("-")[1]

                    if current_cat != token_cat:
                        # Clearing entity
                        new_entity = "{:d}_{}_{}".format(i, current_cat,
                                                         "-".join([str(item) for item in sorted(current_tokens)]))
                        entities.append(new_entity)
                        current_cat = None
                        current_tokens.clear()

                        # Starting new entity
                        current_cat = tok[entity_type].split("-")[1]
                        previous_label = "B"
                        current_tokens.append(j)

                    else:

                        previous_label = "I"
                        current_tokens.append(j)

                elif previous_label in ["I", "E"]:

                    token_cat = tok[entity_type].split("-")[1]

                    if current_cat != token_cat:
                        # Clearing entity
                        new_entity = "{:d}_{}_{}".format(i, current_cat,
                                                         "-".join([str(item) for item in sorted(current_tokens)]))
                        entities.append(new_entity)
                        current_cat = None
                        current_tokens.clear()

                        # Starting new entity
                        current_cat = tok[entity_type].split("-")[1]
                        previous_label = "B"
                        current_tokens.append(j)

                    else:

                        previous_label = "I"
                        current_tokens.append(j)

        if len(current_tokens) > 0:
            # Clearing entity
            new_entity = "{:d}_{}_{}".format(i, current_cat,
                                             "-".join([str(item) for item in sorted(current_tokens)]))
            entities.append(new_entity)

    return entities


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
