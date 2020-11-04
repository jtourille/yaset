import json
import os
from collections import defaultdict

import numpy as np
from tensorboardX import SummaryWriter


class TrainLogger:
    def __init__(self, tensorboard_path: str = None):

        self.tensorboard_path = tensorboard_path
        if self.tensorboard_path is not None:
            self.writer = SummaryWriter(self.tensorboard_path)

        self.losses = defaultdict(dict)

        self.dev_scores = dict()
        self.dev_other_scores = defaultdict(dict)
        self.dev_gs_values = dict()
        self.dev_pred_values = dict()

        self.checkpoints = defaultdict()

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

        self.losses[loss_name][global_step] = loss_value

    def get_loss(self, loss_name: str = None, global_step: int = None):

        return self.losses[loss_name][global_step]

    def add_checkpoint(
        self, step: int = None, checkpoint_payload: dict = None
    ):

        self.checkpoints[step] = checkpoint_payload

    def get_last_checkpoint_string(self, step: int = None):

        str_chunks = list()
        str_chunks.append("step={}".format(step))

        for k, v in self.checkpoints[step].items():
            if type(v) is int:
                str_chunks.append("{}={}".format(k, v))
            else:
                str_chunks.append("{}={:7.5f}".format(k, v))

        return " | ".join(str_chunks)

    def add_dev_score(self, step: int = None, payload: dict = None):

        self.dev_scores[step] = payload

    def add_step_values(
        self,
        step: int = None,
        gs_values: list = None,
        pred_values: list = None,
    ):

        self.dev_gs_values[step] = gs_values
        self.dev_pred_values[step] = pred_values

    def get_step_values(self, step: int = None):

        return self.dev_gs_values[step], self.dev_pred_values[step]

    def get_dev_score(self, step: int = None):

        return self.dev_scores[step]

    def do_early_stopping(self, nb_steps: int = None):

        last_step = sorted(list(self.dev_scores.keys()))[-1]

        best_step, _ = self.get_best_step()

        if last_step - best_step >= nb_steps:
            return True
        else:
            return False

    def get_best_step(self, criterion: str = "f1", reverse: bool = False):

        if not reverse:
            best_step = 0
            best_value = 0.0

            for idx, payload in sorted(self.dev_scores.items()):
                if payload.get("step").get(criterion) > best_value:
                    best_step = idx
                    best_value = payload.get("step").get(criterion)

        else:
            best_step = 0
            best_value = np.inf

            for idx, payload in sorted(self.dev_scores.items()):
                if payload.get("step").get(criterion) < best_value:
                    best_step = idx
                    best_value = payload.get("step").get(criterion)

        return best_step, best_value

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
            "dev_gs_values": self.dev_gs_values,
            "dev_pred_values": self.dev_pred_values,
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
        self.dev_gs_values = {
            int(k): v for k, v in payload["dev_gs_values"].items()
        }
        self.dev_pred_values = {
            int(k): v for k, v in payload["dev_pred_values"].items()
        }
        self.checkpoints = payload["checkpoint"]

    def close_writer(self):

        self.writer.close()
