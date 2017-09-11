import json
import os

from prettytable import PrettyTable


class TrainLogger:

    def __init__(self):

        self.scores = dict()

    def __contains__(self, item):

        return item in self.scores

    def get_best_iteration(self):

        best_iteration = None
        best_score = None

        for ite, payload in self.scores.items():
            if payload["score"] > best_score:
                best_iteration = ite

        return best_iteration

    def add_score(self, ite, score):

        self.scores[ite] = {
            "score": score
        }

    def save_to_file(self, filename):

        json.dump(self.scores, open(os.path.abspath(filename), "w", encoding="UTF-8"))

    def check_patience(self, patience):

        current_iteration = len(self.scores) - 1

        score_list = [i["score"] for k, i in sorted(self.scores.items())]
        score_max = max(score_list)

        best_iteration = score_list.index(score_max)

        if current_iteration - best_iteration >= patience:
            return True
        else:
            return False

    def get_score_table(self):

        x = PrettyTable()

        for iter_nb, payload in sorted(self.scores.items()):
            x.field_names = ["Iteration", "acc."]

            current_iter_nb = "{:3d}".format(iter_nb)
            current_score = "{:.3f}".format(payload["score"])

            x.add_row([current_iter_nb, current_score])

        return x

    def get_removable_iterations(self):

        score_list = [i["score"] for k, i in sorted(self.scores.items())]
        score_max = max(score_list)

        best_iteration = score_list.index(score_max) + 1

        return [i for i in sorted(self.scores) if i != best_iteration]
