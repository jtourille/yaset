import logging
import os
from typing import Callable

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from seqeval.metrics import classification_report
from yaset.utils.logging import TrainLogger

try:
    from apex import amp
except ImportError:
    logging.warning(
        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
    )


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


# def compute_steps(dataset_len: int = None, step: float = 0.05):
#     """
#     Compute step for checkpoint recording and logging
#
#     Args:
#         dataset_len (int): dataset length (i.e. number of training instances)
#         step (float): desired logging step (e.g. 0.05 for logging every 5%)
#
#     Returns:
#         list: steps where logging should occur
#     """
#
#     step = math.ceil(dataset_len * step)
#     steps = list()
#
#     current = 0
#     while current + step <= dataset_len:
#         steps.append(current + step)
#         current += step
#
#     steps.append(dataset_len)
#
#     return steps
#


class Trainer:
    def __init__(
        self,
        accumulation_steps: int = None,
        batch_size: int = None,
        clip_grad_norm: float = None,
        cuda: bool = False,
        dataloader_train: torch.utils.data.DataLoader = None,
        dataloader_dev: torch.utils.data.DataLoader = None,
        eval_function: Callable = None,
        eval_every_n_steps: int = None,
        fp16: bool = None,
        len_dataset_train: int = None,
        len_dataset_dev: int = None,
        log_to_stdout_every_n_step: int = None,
        lr_scheduler: object = None,
        max_steps: int = None,
        model: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        train_logger: TrainLogger = None,
        warmup_scheduler: torch.optim.lr_scheduler.LambdaLR = None,
        working_dir: str = None,
    ):

        self.accumulation_steps = accumulation_steps
        self.batch_size = batch_size
        self.clip_grad_norm = clip_grad_norm
        self.cuda = cuda
        self.dataloader_train = dataloader_train
        self.dataloader_dev = dataloader_dev
        self.eval_function = eval_function
        self.eval_every_n_steps = eval_every_n_steps
        self.fp16 = fp16
        self.log_to_stdout_every_n_step = log_to_stdout_every_n_step
        self.len_dataset_train = len_dataset_train
        self.len_dataset_dev = len_dataset_dev
        self.max_steps = max_steps
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_logger = train_logger
        self.warmup_scheduler = warmup_scheduler
        self.working_dir = working_dir

        self.model_parameter_target_dir = os.path.join(
            self.working_dir, "models"
        )
        if not os.path.isdir(self.model_parameter_target_dir):
            os.makedirs(self.model_parameter_target_dir)

    def perform_training(self):

        step_counter = 0
        processed = 0

        self.optimizer.zero_grad()  # Reinitializing optimizer

        dataloader_iterator = iter(cycle(self.dataloader_train))

        for i in range(self.max_steps * self.accumulation_steps):
            self.model.train()  # Switching to train mode
            batch = next(dataloader_iterator)  # Grabbing next document

            loss, loss_name = self.model.get_loss(batch, cuda=self.cuda)
            loss = loss / self.accumulation_steps

            if self.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Logging loss
            self.train_logger.add_scalar(
                name=loss_name, value=loss.item(), global_step=i + 1
            )

            self.train_logger.add_loss(
                loss_name=loss_name,
                loss_value=loss.item(),
                global_step=i + 1,
            )

            processed += batch.get("labels").size(0)

            if (i + 1) % self.accumulation_steps == 0:
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

                if self.warmup_scheduler is not None:
                    self.warmup_scheduler.step()

                    # Logging to file and stdout when applicable
                if (step_counter + 1) % self.log_to_stdout_every_n_step == 0:
                    checkpoint_payload = {
                        "processed": processed,
                    }
                    for loss_name in self.train_logger.losses:
                        checkpoint_payload[
                            loss_name
                        ] = self.train_logger.get_loss(
                            loss_name=loss_name, global_step=i + 1
                        )

                    self.train_logger.add_checkpoint(
                        step=step_counter + 1,
                        checkpoint_payload=checkpoint_payload,
                    )

                    logging.info(
                        self.train_logger.get_last_checkpoint_string(
                            step_counter + 1
                        )
                    )

                if (
                    (step_counter + 1) % self.eval_every_n_steps == 0
                    or step_counter + 1 == self.max_steps
                ):
                    logging.info(
                        "BEGIN EVALUATION AT STEP: {}".format(step_counter + 1)
                    )
                    logging.info("Gathering evaluation metrics...")

                    gs_values, pred_values = self.test_on_dev(
                        step_counter=step_counter
                    )

                    step_scores = self.train_logger.get_dev_score(
                        step=step_counter + 1
                    )

                    self.train_logger.add_step_values(
                        step=step_counter + 1,
                        gs_values=gs_values,
                        pred_values=pred_values,
                    )

                    logging.info(
                        "STEP #{} | precision={:.3f} | recall={:.3f} | f1={:.3f}".format(
                            step_counter + 1,
                            step_scores["step"]["precision"],
                            step_scores["step"]["recall"],
                            step_scores["step"]["f1_score"],
                        )
                    )

                    logging.info(
                        "\n{}".format(
                            classification_report(gs_values, pred_values)
                        )
                    )

                best_step, _ = self.train_logger.get_best_step(
                    criterion="f1_score"
                )
                # Saving current parameters if this iteration gives the best score on the development dataset
                if best_step == step_counter + 1:
                    logging.info(
                        "Best model so far, saving parameters to disk."
                    )
                    self.clear_model_dir(
                        self.model_parameter_target_dir
                    )  # Clearing parameter directory

                    target_model_filepath = os.path.join(
                        self.model_parameter_target_dir,
                        "model-{}.pth".format(best_step),
                    )
                    torch.save(
                        self.model.state_dict(), target_model_filepath
                    )  # Saving model parameters

                step_counter += 1

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

    def test_on_dev(self, step_counter: int = None):

        self.model.eval()

        with torch.no_grad():

            eval_payload = list()

            for batch in self.dataloader_dev:
                batch_eval_payload = self.model.get_labels(batch, self.cuda)
                eval_payload.append(batch_eval_payload)

            metric_payload = self.eval_function(eval_payload=eval_payload)

            self.train_logger.add_dev_score(
                step=step_counter + 1, payload=metric_payload
            )

        gs = list()
        pred = list()

        for item in eval_payload:
            pred.extend(item.get("pred"))
            gs.extend(item.get("gs"))

        return gs, pred

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

    # def perform_training_old(self):
    #     """
    #     Perform training of a model
    #
    #     Args:
    #         **kwargs: other arguments that will be forwarded to the custom functions
    #
    #     Returns:
    #         None
    #     """
    #
    #     # Creating directory where model parameters will be stored
    #     model_parameter_target_dir = os.path.join(self.working_dir, "models")
    #     ensure_dir(model_parameter_target_dir)
    #
    #     # Training outer-loop
    #     for idx_iteration in range(1, self.max_iterations + 1):
    #         logging.info(
    #             "== BEGIN TRAINING ITERATION {:04d} ==".format(idx_iteration)
    #         )
    #
    #         # Train for one iteration
    #         self.do_one_iteration(idx_iteration=idx_iteration)
    #
    #         logging.info(
    #             "== END TRAINING ITERATION {:04d} ==".format(idx_iteration)
    #         )
    #
    #         logging.info(
    #             "== BEGIN TESTING ON DEV FOR ITERATION {:04d} ==".format(
    #                 idx_iteration
    #             )
    #         )
    #
    #         # Test on development dataset
    #         self.test_on_dev(idx_iteration=idx_iteration)
    #
    #         logging.info(
    #             "== END TESTING ON DEV FOR ITERATION {:04d} ==".format(
    #                 idx_iteration
    #             )
    #         )
    #
    #         # Fetching best iteration index
    #         best_ite, _ = self.train_logger.get_best_iteration()
    #
    #         # Saving current parameters if this iteration gives the best score on the development dataset
    #         if best_ite == idx_iteration:
    #             logging.info("Best model so far, saving parameters to disk.")
    #             self.clear_model_dir(
    #                 model_parameter_target_dir
    #             )  # Clearing parameter directory
    #
    #             target_model_filepath = os.path.join(
    #                 model_parameter_target_dir,
    #                 "model-{}.pth".format(idx_iteration),
    #             )
    #             torch.save(
    #                 self.model.state_dict(), target_model_filepath
    #             )  # Saving model parameters
    #
    #         # Exiting training outer-loop when reaching early stopping condition
    #         if self.train_logger.do_early_stopping(
    #             nb_iterations=self.patience
    #         ):
    #             logging.info(
    #                 "Activating early stopping (current iteration={:03d}, best iteration={:03d})".format(
    #                     idx_iteration,
    #                     best_ite,
    #                 )
    #             )
    #             logging.info(
    #                 "Main score: {}".format(
    #                     self.train_logger.dev_scores[best_ite]
    #                 )
    #             )
    #             for name, value in self.train_logger.dev_other_scores[
    #                 best_ite
    #             ].items():
    #                 logging.info("{} = {}".format(name, value))
    #             break
    #
    #     # Dumping training log values to disk
    #     logging.info("Dumping log object to disk")
    #     target_log_file = os.path.join(
    #         os.path.abspath(self.working_dir), "train_log.pkl"
    #     )
    #     target_tensorboard_log_file = os.path.join(
    #         os.path.abspath(self.working_dir), "tb_log.pkl"
    #     )
    #     self.train_logger.dump_to_disk(
    #         custom_log_file=target_log_file,
    #         tensorboard_log_file=target_tensorboard_log_file,
    #     )
    #     self.train_logger.close_writer()
