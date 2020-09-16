import logging
import math
import os
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

from .logging import TrainLogger
from ..utils.path import ensure_dir

try:
    from apex import amp
except ImportError:
    logging.warning(
        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
    )


def compute_steps(dataset_len: int = None, step: float = 0.05):
    """
    Compute step for checkpoint recording and logging

    Args:
        dataset_len (int): dataset length (i.e. number of training instances)
        step (float): desired logging step (e.g. 0.05 for logging every 5%)

    Returns:
        list: steps where logging should occur
    """

    step = math.ceil(dataset_len * step)
    steps = list()

    current = 0
    while current + step <= dataset_len:
        steps.append(current + step)
        current += step

    steps.append(dataset_len)

    return steps


class Trainer:
    def __init__(
        self,
        accumulation_steps: int = None,
        clip_grad_norm: float = None,
        cuda: bool = False,
        fp16: bool = None,
        dataloader_train: torch.utils.data.DataLoader = None,
        dataloader_dev: torch.utils.data.DataLoader = None,
        eval_function: Callable = None,
        len_dataset_train: int = None,
        len_dataset_dev: int = None,
        log_to_stdout_step: float = 0.05,
        max_iterations: int = 100,
        model: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        patience: int = 10,
        scheduler: object = None,
        train_logger: TrainLogger = None,
        working_dir: str = None,
    ):

        self.accumulation_steps = accumulation_steps
        self.clip_grad_norm = clip_grad_norm
        self.cuda = cuda
        self.fp16 = fp16
        self.dataloader_train = dataloader_train
        self.dataloader_dev = dataloader_dev
        self.eval_function = eval_function
        self.len_dataset_train = len_dataset_train
        self.len_dataset_dev = len_dataset_dev
        self.log_to_stdout_step = log_to_stdout_step
        self.max_iterations = max_iterations
        self.model = model
        self.optimizer = optimizer
        self.patience = patience
        self.scheduler = scheduler
        self.train_logger = train_logger
        self.working_dir = working_dir

        self.global_step = 0

    def do_one_iteration(self, idx_iteration: int = None):

        self.model.train()  # Switching to train mode

        processed_iteration = 0
        processed_batch_checkpoint = 0

        logging_steps = compute_steps(
            self.len_dataset_train, step=self.log_to_stdout_step
        )
        # todo: check if there is something to print

        # Reinitializing optimizer
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.dataloader_train):
            # Incrementing counters
            processed_iteration += batch["size"]
            self.global_step += batch["size"]
            processed_batch_checkpoint += 1

            # Forward pass
            loss, loss_name = self.model.get_loss(batch, cuda=self.cuda)
            loss = loss / self.accumulation_steps

            if self.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Logging loss
            self.train_logger.add_scalar(
                name=loss_name, value=loss.item(), global_step=self.global_step
            )

            self.train_logger.add_loss(
                loss_name=loss_name,
                loss_value=loss.item(),
                global_step=self.global_step,
            )

            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Clipping gradient if necessary
                if self.clip_grad_norm is not None:
                    if self.fp16:
                        nn.utils.clip_grad_norm_(
                            amp.master_params(self.optimizer),
                            self.clip_grad_norm,
                        )
                    else:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.clip_grad_norm,
                            norm_type=2,
                        )

                # Performing optimization step
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Logging gradient mean and std
            # for param_name, param in self.model.named_parameters():
            #     if param.grad is None:
            #         continue
            #
            #     values = param.grad.clone().cpu().data.numpy()
            #     values = values[~np.isnan(values)]
            #
            #     self.train_logger.add_scalar(
            #         name="grad_mean/" + param_name,
            #         value=values.mean(),
            #         global_step=self.global_step,
            #     )
            #     self.train_logger.add_scalar(
            #         name="grad_std/" + param_name,
            #         value=values.std(),
            #         global_step=self.global_step,
            #     )

            if (
                processed_iteration >= logging_steps[0]
                or processed_iteration == self.len_dataset_train
            ):
                # for param_name, param in self.model.named_parameters():
                #
                #     values = param.clone().cpu().data.numpy()
                #     values = values[~np.isnan(values)]
                #
                #     self.train_logger.add_scalar(
                #         name="parameter_mean/" + param_name,
                #         value=values.mean(),
                #         global_step=self.global_step,
                #     )
                #     self.train_logger.add_scalar(
                #         name="parameter_std/" + param_name,
                #         value=values.std(),
                #         global_step=self.global_step,
                #     )

                checkpoint_name = "{:6.2f}".format(
                    (processed_iteration / self.len_dataset_train) * 100
                )
                _ = logging_steps.pop(0)
                checkpoint_payload = {"processed": processed_iteration}
                for loss_name in self.train_logger.losses:
                    checkpoint_payload[loss_name] = np.mean(
                        [
                            loss_value
                            for _, loss_value in self.train_logger.losses[
                                loss_name
                            ]
                        ]
                    )

                self.train_logger.add_checkpoint(
                    idx_iteration=idx_iteration,
                    checkpoint_name=checkpoint_name,
                    checkpoint_payload=checkpoint_payload,
                )

                logging.info(
                    self.train_logger.get_last_checkpoint_string(idx_iteration)
                )

                processed_batch_checkpoint = 0

        # for name, param in self.model.named_parameters():
        #     # Removing NaN values before making a histogram
        #     values = param.clone().cpu().data.numpy()
        #     values = values[~np.isnan(values)]
        #
        #     self.train_logger.add_histogram("params/" + name, values, idx_iteration)

    def test_on_dev(self, idx_iteration: int = None):

        with torch.no_grad():

            self.model.eval()

            processed_iteration = 0

            steps = compute_steps(
                self.len_dataset_dev, step=self.log_to_stdout_step
            )

            eval_payload = list()

            for batch_idx, batch in enumerate(self.dataloader_dev):

                processed_iteration += batch["size"]

                batch_eval_payload = self.model.get_labels(
                    batch, self.cuda, idx_iteration=idx_iteration
                )
                eval_payload.append(batch_eval_payload)

                if (
                    processed_iteration >= steps[0]
                    or processed_iteration == self.len_dataset_dev
                ):
                    _ = steps.pop(0)
                    logging.info(
                        "Processed={:5d} ({:6.2f}%)".format(
                            processed_iteration,
                            (processed_iteration / self.len_dataset_dev) * 100,
                        )
                    )

            eval_payload = self.eval_function(eval_payload=eval_payload)
            if self.scheduler is not None:
                self.scheduler.step(eval_payload["main"])

            self.train_logger.add_dev_score(
                idx_iteration=idx_iteration, dev_score=eval_payload["main"]
            )
            for name, value in eval_payload["tensorboard"]:
                self.train_logger.add_scalar(
                    name=name, value=value, global_step=idx_iteration
                )
                self.train_logger.add_other_score_dev(
                    idx_iteration=idx_iteration,
                    score_name=name,
                    score_value=value,
                )

            logging.info("Iteration score: {}".format(eval_payload["main"]))
            for name, value in self.train_logger.dev_other_scores[
                idx_iteration
            ].items():
                logging.info("{} = {}".format(name, value))

    @staticmethod
    def clear_model_dir(model_dir):
        """
        Remove old model parameter files

        Args:
            model_dir (str): model parameter directory

        Returns:
            None
        """

        for root, dirs, files in os.walk(model_dir):
            for filename in files:
                file_to_be_removed = os.path.join(root, filename)
                os.remove(file_to_be_removed)

    def perform_training(self):
        """
        Perform training of a model

        Args:
            **kwargs: other arguments that will be forwarded to the custom functions

        Returns:
            None
        """

        # Creating directory where model parameters will be stored
        model_parameter_target_dir = os.path.join(self.working_dir, "models")
        ensure_dir(model_parameter_target_dir)

        # Training outer-loop
        for idx_iteration in range(1, self.max_iterations + 1):
            logging.info(
                "== BEGIN TRAINING ITERATION {:04d} ==".format(idx_iteration)
            )

            # Train for one iteration
            self.do_one_iteration(idx_iteration=idx_iteration)

            logging.info(
                "== END TRAINING ITERATION {:04d} ==".format(idx_iteration)
            )

            logging.info(
                "== BEGIN TESTING ON DEV FOR ITERATION {:04d} ==".format(
                    idx_iteration
                )
            )

            # Test on development dataset
            self.test_on_dev(idx_iteration=idx_iteration)

            logging.info(
                "== END TESTING ON DEV FOR ITERATION {:04d} ==".format(
                    idx_iteration
                )
            )

            # Fetching best iteration index
            best_ite, _ = self.train_logger.get_best_iteration()

            # Saving current parameters if this iteration gives the best score on the development dataset
            if best_ite == idx_iteration:
                logging.info("Best model so far, saving parameters to disk.")
                self.clear_model_dir(
                    model_parameter_target_dir
                )  # Clearing parameter directory

                target_model_filepath = os.path.join(
                    model_parameter_target_dir,
                    "model-{}.pth".format(idx_iteration),
                )
                torch.save(
                    self.model.state_dict(), target_model_filepath
                )  # Saving model parameters

            # Exiting training outer-loop when reaching early stopping condition
            if self.train_logger.do_early_stopping(
                nb_iterations=self.patience
            ):
                logging.info(
                    "Activating early stopping (current iteration={:03d}, best iteration={:03d})".format(
                        idx_iteration,
                        best_ite,
                    )
                )
                logging.info(
                    "Main score: {}".format(
                        self.train_logger.dev_scores[best_ite]
                    )
                )
                for name, value in self.train_logger.dev_other_scores[
                    best_ite
                ].items():
                    logging.info("{} = {}".format(name, value))
                break

        # Dumping training log values to disk
        logging.info("Dumping log object to disk")
        target_log_file = os.path.join(
            os.path.abspath(self.working_dir), "train_log.pkl"
        )
        target_tensorboard_log_file = os.path.join(
            os.path.abspath(self.working_dir), "tb_log.pkl"
        )
        self.train_logger.dump_to_disk(
            custom_log_file=target_log_file,
            tensorboard_log_file=target_tensorboard_log_file,
        )
        self.train_logger.close_writer()
