import json
import logging
import os
import shutil
from typing import Tuple, Dict

import joblib
import torch
from torch.utils.data import DataLoader

from .nn.crf import allowed_transitions
from .nn.embedding import Embedder
from .nn.lstmcrf import LSTMCRF
from .utils.config import replace_auto
from .utils.data import NERDataset, collate_ner
from .utils.eval import eval_ner
from .utils.logging import TrainLogger
from .utils.mapping import extract_mappings_and_pretrained_matrix
from .utils.training import Trainer
from .utils.copy import copy_embedding_models

import _jsonnet


def create_dataloader(mappings: Dict = None,
                      options: Dict = None,
                      instance_json_file: str = None,
                      test: bool = False) -> Tuple[DataLoader, int]:

    # Building datasets (NER, ATT and REL)
    ner_dataset = NERDataset(mappings=mappings,
                             instance_json_file=os.path.abspath(instance_json_file),
                             testing=options.get("testing"))

    logging.info("Dataset size: {}".format(len(ner_dataset)))

    if test:
        batch_size = options.get("training").get("test_batch_size")
    else:
        batch_size = options.get("training").get("train_batch_size")

    logging.info("Batch size: {}".format(batch_size))

    ner_dataloader = DataLoader(ner_dataset,
                                num_workers=options.get("training").get("num_dataloader_workers"),
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=lambda b: collate_ner(
                                    b,
                                    tok_pad_id=mappings["tokens"].get("<pad>"),
                                    chr_pad_id=mappings["characters"].get("<pad>"),
                                    options=options))

    return ner_dataloader, len(ner_dataset)


def train_model(option_file: str = None,
                output_dir: str = None):

    target_json_file = os.path.join(os.path.abspath(output_dir), "options.jsonnet")
    shutil.copy(os.path.abspath(option_file), target_json_file)

    options_json = _jsonnet.evaluate_file(option_file)
    options = json.loads(options_json)

    replace_auto(options=options)

    mappings, pretrained_matrix = extract_mappings_and_pretrained_matrix(options=options,
                                                                         output_dir=output_dir)

    copy_embedding_models(embeddings_options=options.get("embeddings"),
                          output_dir=output_dir)

    pretrained_matrix_size = None

    if pretrained_matrix is not None:
        target_pretrained_matrix_file = os.path.join(os.path.abspath(output_dir), "pretrained_matrix.pkl")
        logging.debug("Dumping pretrained matrix to disk: {}".format(target_pretrained_matrix_file))
        joblib.dump(pretrained_matrix, target_pretrained_matrix_file)

        target_pretrained_matrix_size_file = os.path.join(os.path.abspath(output_dir), "pretrained_matrix_size.json")
        logging.debug("Dumping pretrained matrix size to disk: {}".format(target_pretrained_matrix_size_file))
        with open(target_pretrained_matrix_size_file, "w", encoding="UTF-8") as output_file:
            json.dump(pretrained_matrix.shape, output_file)

        pretrained_matrix_size = pretrained_matrix.shape

    target_mapping_file = os.path.join(os.path.abspath(output_dir), "mappings.json")
    logging.debug("Dumping mappings to disk: {}".format(target_mapping_file))
    with open(target_mapping_file, "w", encoding="UTF-8") as output_file:
        json.dump(mappings, output_file)

    torch.set_num_threads(options.get("training").get("num_global_workers"))

    logging.debug("Creating logger")
    target_tensorboard_path = os.path.join(output_dir, "tb")
    train_logger = TrainLogger(tensorboard_path=target_tensorboard_path)

    logging.info("Creating dataloaders")
    dataloader_train, len_train = create_dataloader(mappings=mappings,
                                                    options=options,
                                                    instance_json_file=options.get("data").get("train_file"),
                                                    test=False)

    dataloader_dev, len_dev = create_dataloader(mappings=mappings,
                                                options=options,
                                                instance_json_file=options.get("data").get("dev_file"),
                                                test=True)

    logging.info("Building model")
    embedder = Embedder(embeddings_options=options.get("embeddings"),
                        pretrained_matrix=pretrained_matrix,
                        pretrained_matrix_size=pretrained_matrix_size,
                        mappings=mappings,
                        embedding_root_dir=output_dir)

    constraints = allowed_transitions(options.get("data").get("format"),
                                      {v: k for k, v in mappings["ner_labels"].items()})

    model = LSTMCRF(embedder=embedder,
                    constraints=constraints,
                    ffnn_hidden_layer_use=options.get("network_structure").get("ffnn").get("use"),
                    ffnn_hidden_layer_size=options.get("network_structure").get("ffnn").get("hidden_layer_size"),
                    ffnn_activation_function=options.get("network_structure").get("ffnn").get("activation_function"),
                    ffnn_input_dropout_rate=options.get("training").get("ffnn_input_dropout_rate"),
                    input_size=embedder.embedding_size,
                    input_dropout_rate=options.get("training").get("input_dropout_rate"),
                    lstm_cell_size=options.get("network_structure").get("cell_size"),
                    lstm_hidden_size=options.get("network_structure").get("hidden_size"),
                    lstm_layer_dropout_rate=options.get("training").get("lstm_layer_dropout_rate"),
                    mappings=mappings,
                    nb_layers=options.get("network_structure").get("nb_layers"),
                    num_labels=len(mappings["ner_labels"]),
                    skip_connections=options.get("network_structure").get("skip_connections"))

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=options.get("training").get("lr_rate"))

    if options.get("training").get("lr_scheduler").get("use"):
        logging.debug("Activating learning rate scheduling")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=options.get("training").get("lr_scheduler").get("mode"),
            factor=options.get("training").get("lr_scheduler").get("factor"),
            patience=options.get("training").get("lr_scheduler").get("patience"),
            verbose=options.get("training").get("lr_scheduler").get("verbose"),
            threshold=options.get("training").get("lr_scheduler").get("threshold"),
            threshold_mode=options.get("training").get("lr_scheduler").get("threshold_mode")
        )
    else:
        scheduler = None

    if options.get("training").get("cuda"):
        logging.info("Switching to cuda")
        model.cuda()

    trainer = Trainer(clip_grad_norm=options.get("training").get("clip_grad_norm"),
                      cuda=options.get("training").get("cuda"),
                      dataloader_train=dataloader_train,
                      dataloader_dev=dataloader_dev,
                      eval_function=eval_ner,
                      len_dataset_train=len_train,
                      len_dataset_dev=len_dev,
                      log_to_stdout_step=0.05,
                      mappings=mappings,
                      max_iterations=options.get("training").get("max_iterations"),
                      model=model,
                      optimizer=optimizer,
                      patience=options.get("training").get("patience"),
                      scheduler=scheduler,
                      train_logger=train_logger,
                      working_dir=os.path.abspath(output_dir))

    trainer.perform_training()

