import json
import os
from collections import defaultdict

from tensorboardX import SummaryWriter


class TrainLogger:
    def __init__(self, tensorboard_path: str = None):

        self.tensorboard_path = os.path.abspath(tensorboard_path)
        self.writer = SummaryWriter(self.tensorboard_path)

        self.losses = defaultdict(list)
        self.dev_scores = dict()
        self.test_scores = dict()
        self.dev_other_scores = defaultdict(dict)
        self.test_other_scores = defaultdict(dict)
        self.checkpoints = defaultdict(dict)

    def add_scalar(
        self, name: str = None, value: float = None, global_step: int = None
    ):

        self.writer.add_scalar(name, value, global_step=global_step)

    def add_histogram(
        self,
        name: str = None,
        value: str = None,
        global_step: int = None,
        bins: str = "auto",
    ):

        self.writer.add_histogram(
            name, value, global_step=global_step, bins=bins
        )

    def add_loss(
        self,
        loss_value: float = None,
        loss_name: str = None,
        global_step: int = None,
    ):

        self.losses[loss_name].append((int(global_step), loss_value))

    def add_checkpoint(
        self,
        idx_iteration: int = None,
        checkpoint_name: str = None,
        checkpoint_payload: dict = None,
    ):

        self.checkpoints[idx_iteration][checkpoint_name] = checkpoint_payload

    def get_last_checkpoint_string(self, idx_iteration: int = None):

        last_checkpoint_id = list(self.checkpoints[idx_iteration].keys())[-1]

        str_chunks = list()
        str_chunks.append("{}".format(last_checkpoint_id))

        for k, v in self.checkpoints[idx_iteration][
            last_checkpoint_id
        ].items():
            if type(v) is int:
                str_chunks.append("{}={}".format(k, v))
            else:
                str_chunks.append("{}={:7.5f}".format(k, v))

        return " | ".join(str_chunks)

    def add_dev_score(self, idx_iteration: int = None, dev_score: float = 0.0):

        self.dev_scores[idx_iteration] = dev_score

    def do_early_stopping(self, nb_iterations: int = None):

        last_iteration_idx = list(self.dev_scores.keys())[-1]

        best_ite_idx, _ = self.get_best_iteration()

        if last_iteration_idx - best_ite_idx >= nb_iterations:
            return True
        else:
            return False

    def get_best_iteration(self):

        best_idx = 0
        best_value = 0.0

        for idx, score in self.dev_scores.items():
            if score >= best_value:
                best_idx = idx
                best_value = score

        return best_idx, best_value

    def add_other_score_dev(
        self,
        idx_iteration: int = None,
        score_name: str = None,
        score_value: float = None,
    ):

        self.dev_other_scores[idx_iteration][score_name] = score_value

    def dump_to_disk(
        self, custom_log_file: str = None, tensorboard_log_file: str = None
    ):

        payload = {
            "losses": self.losses,
            "dev_scores": self.dev_scores,
            "dev_other_scores": self.dev_other_scores,
            "checkpoint": self.checkpoints,
        }

        with open(
            os.path.abspath(custom_log_file), "w", encoding="UTF-8"
        ) as output_file:
            json.dump(payload, output_file)

        self.writer.export_scalars_to_json(tensorboard_log_file)

    def load_json_file(self, filepath: str = None):

        with open(
            os.path.abspath(filepath), "r", encoding="UTF-8"
        ) as input_file:
            payload = json.load(input_file)

        self.losses = payload["losses"]
        self.dev_scores = {int(k): v for k, v in payload["dev_scores"].items()}
        self.dev_other_scores = {
            int(k): v for k, v in payload["dev_other_scores"].items()
        }
        self.checkpoints = payload["checkpoint"]

    def close_writer(self):

        self.writer.close()
