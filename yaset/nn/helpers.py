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
