import json
import os

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

    def get_best_iteration(self):
        """
        Return the iteration ID of the best iteration
        :return: iteration ID (int)
        """

        best_iteration = None
        best_score = None

        for ite, payload in self.iterations_log.items():
            if payload["dev_score"] > best_score or best_score is None:
                best_iteration = ite
                best_score = payload["dev_score"]

        return best_iteration

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

        if current_iteration - best_iteration >= patience:
            return True
        else:
            return False

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


def conll_eval(sequences):

    source_entity = get_entities(sequences, "gs")
    pred_entity = get_entities(sequences, "pred")

    pred = 0
    corr = 0
    gs = 0

    for entity in source_entity:
        if entity in pred_entity:
            corr += 1

    pred += len(pred_entity)
    gs += len(source_entity)

    try:
        precision = float(corr) / pred
        recall = float(corr) / gs
        f1 = (2 * precision * recall) / (precision + recall)
    except Exception:
        precision = 0.0
        recall = 0.0
        f1 = 0.0

    return precision, recall, f1


def get_entities(sequences, att):

    entities = list()

    for i, sequence in enumerate(sequences):

        previous_label = "O"
        current_cat = None
        current_tokens = list()

        for j, tok in enumerate(sequence):
            if tok[att].startswith("B"):

                if previous_label == "I":

                    # Clearing entity
                    new_entity = "{:d}_{}_{}".format(i, current_cat,
                                                     "-".join([str(item) for item in sorted(current_tokens)]))
                    entities.append(new_entity)
                    current_cat = None
                    current_tokens.clear()

                    # Starting new entity
                    current_cat = tok[att].split("-")[1]
                    previous_label = "B"
                    current_tokens.append(j)

                elif previous_label == "B":

                    # Clearing entity
                    new_entity = "{:d}_{}_{}".format(i, current_cat,
                                                     "-".join([str(item) for item in sorted(current_tokens)]))
                    entities.append(new_entity)
                    current_cat = None
                    current_tokens.clear()

                    # Starting new entity
                    current_cat = tok[att].split("-")[1]
                    previous_label = "B"
                    current_tokens.append(j)

                elif previous_label == "O":

                    # Starting new entity
                    current_cat = tok[att].split("-")[1]
                    previous_label = "B"
                    current_tokens.append(j)

            elif tok[att].startswith("O"):

                if previous_label in ["B", "I"]:

                    # Clearing entity
                    new_entity = "{:d}_{}_{}".format(i, current_cat,
                                                     "-".join([str(item) for item in sorted(current_tokens)]))
                    entities.append(new_entity)
                    current_cat = None
                    current_tokens.clear()
                    previous_label = "O"

            elif tok[att].startswith("I"):

                if previous_label == "O":

                    # Starting new entity
                    current_cat = tok[att].split("-")[1]
                    previous_label = "B"
                    current_tokens.append(j)

                elif previous_label == "B":

                    token_cat = tok[att].split("-")[1]

                    if current_cat != token_cat:
                        # Clearing entity
                        new_entity = "{:d}_{}_{}".format(i, current_cat,
                                                         "-".join([str(item) for item in sorted(current_tokens)]))
                        entities.append(new_entity)
                        current_cat = None
                        current_tokens.clear()

                        # Starting new entity
                        current_cat = tok[att].split("-")[1]
                        previous_label = "B"
                        current_tokens.append(j)

                    else:

                        previous_label = "I"
                        current_tokens.append(j)

                elif previous_label == "I":

                    token_cat = tok[att].split("-")[1]

                    if current_cat != token_cat:
                        # Clearing entity
                        new_entity = "{:d}_{}_{}".format(i, current_cat,
                                                         "-".join([str(item) for item in sorted(current_tokens)]))
                        entities.append(new_entity)
                        current_cat = None
                        current_tokens.clear()

                        # Starting new entity
                        current_cat = tok[att].split("-")[1]
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
