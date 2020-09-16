import json
import logging
import os
import shutil
from typing import Tuple, Dict

import _jsonnet
import joblib
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from yaset.nn.crf import allowed_transitions
from yaset.nn.embedding import Embedder
from yaset.nn.lstmcrf import AugmentedLSTMCRF
from yaset.utils.config import replace_auto
from yaset.utils.copy import copy_embedding_models
from yaset.utils.data import NERDataset, collate_ner
from yaset.utils.eval import eval_ner
from yaset.utils.logging import TrainLogger
from yaset.utils.mapping import extract_mappings_and_pretrained_matrix
from yaset.utils.training import Trainer


def create_dataloader(
    mappings: Dict = None,
    options: Dict = None,
    instance_json_file: str = None,
    test: bool = False,
    working_dir: str = None,
) -> Tuple[DataLoader, int]:

    singleton_replacement_ratio = 0.0

    if not test:
        if options.get("embeddings").get("pretrained").get("use"):
            singleton_replacement_ratio = (
                options.get("embeddings")
                .get("pretrained")
                .get("singleton_replacement_ratio")
            )

    bert_use = options.get("embeddings").get("bert").get("use")
    bert_voc_dir = os.path.join(working_dir, "bert", "vocab")
    bert_lowercase = options.get("embeddings").get("bert").get("do_lower_case")

    ner_dataset = NERDataset(
        mappings=mappings,
        instance_json_file=os.path.abspath(instance_json_file),
        testing=options.get("testing"),
        singleton_replacement_ratio=singleton_replacement_ratio,
        bert_use=bert_use,
        bert_voc_dir=bert_voc_dir,
        bert_lowercase=bert_lowercase,
    )

    logging.info("Dataset size: {}".format(len(ner_dataset)))

    if test:
        batch_size = options.get("training").get("test_batch_size")
    else:
        batch_size = options.get("training").get("train_batch_size")

    logging.info("Batch size: {}".format(batch_size))

    if not test:
        drop_last = True
    else:
        drop_last = False

    ner_dataloader = DataLoader(
        ner_dataset,
        num_workers=options.get("training").get("num_dataloader_workers"),
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        collate_fn=lambda b: collate_ner(
            b,
            tok_pad_id=mappings["tokens"].get("<pad>"),
            chr_pad_id_literal=mappings["characters_literal"].get("<pad>"),
            chr_pad_id_utf8=mappings["characters_utf8"].get("<pad>"),
            bert_use=bert_use,
            options=options,
        ),
    )

    return ner_dataloader, len(ner_dataset)


def train_model(option_file: str = None, output_dir: str = None):

    target_json_file = os.path.join(
        os.path.abspath(output_dir), "options.jsonnet"
    )
    shutil.copy(os.path.abspath(option_file), target_json_file)

    options_json = _jsonnet.evaluate_file(option_file)
    options = json.loads(options_json)

    replace_auto(options=options)

    mappings, pretrained_matrix = extract_mappings_and_pretrained_matrix(
        options=options, output_dir=output_dir
    )

    copy_embedding_models(
        embeddings_options=options.get("embeddings"), output_dir=output_dir
    )

    pretrained_matrix_size = None

    if pretrained_matrix is not None:
        target_pretrained_matrix_file = os.path.join(
            os.path.abspath(output_dir), "pretrained_matrix.pkl"
        )
        logging.debug(
            "Dumping pretrained matrix to disk: {}".format(
                target_pretrained_matrix_file
            )
        )
        joblib.dump(pretrained_matrix, target_pretrained_matrix_file)

        target_pretrained_matrix_size_file = os.path.join(
            os.path.abspath(output_dir), "pretrained_matrix_size.json"
        )
        logging.debug(
            "Dumping pretrained matrix size to disk: {}".format(
                target_pretrained_matrix_size_file
            )
        )
        with open(
            target_pretrained_matrix_size_file, "w", encoding="UTF-8"
        ) as output_file:
            json.dump(pretrained_matrix.shape, output_file)

        pretrained_matrix_size = pretrained_matrix.shape

    target_mapping_file = os.path.join(
        os.path.abspath(output_dir), "mappings.json"
    )
    logging.debug("Dumping mappings to disk: {}".format(target_mapping_file))
    with open(target_mapping_file, "w", encoding="UTF-8") as output_file:
        json.dump(mappings, output_file)

    torch.set_num_threads(options.get("training").get("num_global_workers"))

    logging.debug("Creating logger")
    target_tensorboard_path = os.path.join(output_dir, "tb")
    train_logger = TrainLogger(tensorboard_path=target_tensorboard_path)

    logging.info("Creating dataloaders")
    dataloader_train, len_train = create_dataloader(
        mappings=mappings,
        options=options,
        instance_json_file=options.get("data").get("train_file"),
        test=False,
        working_dir=output_dir,
    )

    dataloader_dev, len_dev = create_dataloader(
        mappings=mappings,
        options=options,
        instance_json_file=options.get("data").get("dev_file"),
        test=True,
        working_dir=output_dir,
    )

    logging.info("Building model")
    embedder = Embedder(
        embeddings_options=options.get("embeddings"),
        pretrained_matrix=pretrained_matrix,
        pretrained_matrix_size=pretrained_matrix_size,
        mappings=mappings,
        embedding_root_dir=output_dir,
    )

    constraints = allowed_transitions(
        options.get("data").get("format"),
        {v: k for k, v in mappings["ner_labels"].items()},
    )

    model = AugmentedLSTMCRF(
        embedder=embedder,
        constraints=constraints,
        ffnn_hidden_layer_use=options.get("network_structure")
        .get("ffnn")
        .get("use"),
        ffnn_hidden_layer_size=options.get("network_structure")
        .get("ffnn")
        .get("hidden_layer_size"),
        ffnn_activation_function=options.get("network_structure")
        .get("ffnn")
        .get("activation_function"),
        ffnn_input_dropout_rate=options.get("network_structure")
        .get("ffnn")
        .get("input_dropout_rate"),
        input_size=embedder.embedding_size,
        lstm_input_dropout_rate=options.get("network_structure").get(
            "input_dropout_rate"
        ),
        lstm_hidden_size=options.get("network_structure")
        .get("lstm")
        .get("hidden_size"),
        lstm_layer_dropout_rate=options.get("network_structure")
        .get("lstm")
        .get("layer_dropout_rate"),
        mappings=mappings,
        nb_layers=options.get("network_structure")
        .get("lstm")
        .get("nb_layers"),
        num_labels=len(mappings["ner_labels"]),
        use_highway=options.get("network_structure")
        .get("lstm")
        .get("highway"),
    )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": options.get("training").get("weight_decay"),
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    learning_rate = options.get("training").get("lr_rate")

    if options.get("training").get("optimizer") == "adam":
        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters, lr=learning_rate
        )

    elif options.get("training").get("optimizer") == "adamw":
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    else:
        raise Exception(
            "The optimizer does not exist or is not supported by YASET: {}".format(
                options.get("training").get("optimizer")
            )
        )

    if options.get("training").get("lr_scheduler").get("use"):
        logging.debug("Activating learning rate scheduling")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=options.get("training").get("lr_scheduler").get("mode"),
            factor=options.get("training").get("lr_scheduler").get("factor"),
            patience=options.get("training")
            .get("lr_scheduler")
            .get("patience"),
            verbose=options.get("training").get("lr_scheduler").get("verbose"),
            threshold=options.get("training")
            .get("lr_scheduler")
            .get("threshold"),
            threshold_mode=options.get("training")
            .get("lr_scheduler")
            .get("threshold_mode"),
        )
    else:
        scheduler = None

    if options.get("training").get("cuda"):
        logging.info("Switching to cuda")
        model.cuda()

    if options.get("training").get("fp16"):
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        logging.info(
            "Using fp16: {}".format(options.get("training").get("fp16_level"))
        )
        model, optimizer = amp.initialize(
            model,
            optimizer,
            opt_level=options.get("training").get("fp16_level"),
        )

    trainer = Trainer(
        accumulation_steps=options.get("training").get("accumulation_steps"),
        clip_grad_norm=options.get("training").get("clip_grad_norm"),
        cuda=options.get("training").get("cuda"),
        fp16=options.get("training").get("fp16"),
        dataloader_train=dataloader_train,
        dataloader_dev=dataloader_dev,
        eval_function=eval_ner,
        len_dataset_train=len_train,
        len_dataset_dev=len_dev,
        log_to_stdout_step=0.05,
        max_iterations=options.get("training").get("max_iterations"),
        model=model,
        optimizer=optimizer,
        patience=options.get("training").get("patience"),
        scheduler=scheduler,
        train_logger=train_logger,
        working_dir=os.path.abspath(output_dir),
    )

    trainer.perform_training()
