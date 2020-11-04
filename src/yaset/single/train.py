import json
import logging
import os
import shutil
from typing import Tuple, Dict

import _jsonnet
import joblib
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from yaset.nn.crf import allowed_transitions
from yaset.nn.embedding import Embedder
from yaset.nn.lstmcrf import AugmentedLSTMCRF
from yaset.utils.config import replace_auto
from yaset.utils.copy import copy_embedding_models
from yaset.utils.data import NERDataset, collate_ner
from yaset.utils.eval import eval_ner
from yaset.utils.logging import TrainLogger
from yaset.utils.mapping import (
    extract_mappings_and_pretrained_matrix,
    extract_char_mapping,
    extract_label_mapping,
)
from yaset.utils.training import Trainer

try:
    from apex import amp
except ImportError:
    logging.warning(
        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
    )


def create_dataloader(
    mappings: Dict = None,
    options: Dict = None,
    instance_json_file: str = None,
    test: bool = False,
    working_dir: str = None,
) -> Tuple[DataLoader, int, Dataset]:
    singleton_replacement_ratio = 0.0

    if not test:
        if options.get("embeddings").get("pretrained").get("use"):
            singleton_replacement_ratio = (
                options.get("embeddings")
                .get("pretrained")
                .get("singleton_replacement_ratio")
            )

    bert_use = options.get("embeddings").get("bert").get("use")
    bert_voc_dir = os.path.join(working_dir, "embeddings", "bert", "vocab")
    bert_lowercase = options.get("embeddings").get("bert").get("do_lower_case")

    pretrained_use = options.get("embeddings").get("pretrained").get("use")
    elmo_use = options.get("embeddings").get("pretrained").get("use")
    char_use = options.get("embeddings").get("chr_cnn").get("use")

    ner_dataset = NERDataset(
        mappings=mappings,
        instance_conll_file=os.path.abspath(instance_json_file),
        debug=options.get("debug"),
        singleton_replacement_ratio=singleton_replacement_ratio,
        bert_use=bert_use,
        bert_voc_dir=bert_voc_dir,
        bert_lowercase=bert_lowercase,
        char_use=char_use,
        elmo_use=elmo_use,
        pretrained_use=pretrained_use,
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

    tok_pad_id = None
    chr_pad_id_literal = None
    chr_pad_id_utf8 = None

    if pretrained_use:
        tok_pad_id = mappings.get("toks").get("pad_id")

    if char_use:
        chr_pad_id_literal = (
            mappings.get("chrs").get("char_literal").get("<pad>")
        )
        chr_pad_id_utf8 = mappings.get("chrs").get("char_utf8").get("<pad>")

    ner_dataloader = DataLoader(
        ner_dataset,
        num_workers=options.get("training").get("num_dataloader_workers"),
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        collate_fn=lambda b: collate_ner(
            b,
            tok_pad_id=tok_pad_id,
            chr_pad_id_literal=chr_pad_id_literal,
            chr_pad_id_utf8=chr_pad_id_utf8,
            bert_use=bert_use,
            char_use=char_use,
            elmo_use=elmo_use,
            pretrained_use=pretrained_use,
            options=options,
        ),
    )

    return ner_dataloader, len(ner_dataset), ner_dataset


def train_single_model(
    option_file: str = None, output_dir: str = None
) -> None:
    """
    Train a NER model

    :param option_file: model configuration file (jsonnet format)
    :type option_file: str
    :param output_dir: model output directory
    :type output_dir: str

    :return: None
    """

    target_json_file = os.path.join(
        os.path.abspath(output_dir), "options.jsonnet"
    )
    shutil.copy(os.path.abspath(option_file), target_json_file)

    options_json = _jsonnet.evaluate_file(option_file)
    options = json.loads(options_json)

    replace_auto(options=options)

    if options.get("debug"):
        options["training"]["num_epochs"] = 1

    target_embedding_dir = os.path.join(output_dir, "embeddings")

    copy_embedding_models(
        embeddings_options=options.get("embeddings"),
        output_dir=target_embedding_dir,
    )

    (
        pretrained_matrix,
        pretrained_matrix_size,
        pretrained_matrix_mapping,
    ) = extract_mappings_and_pretrained_matrix(
        options=options, output_dir=output_dir
    )

    char_mapping = extract_char_mapping(
        instance_file=options.get("data").get("train_file")
    )
    label_mapping = extract_label_mapping(
        instance_file=options.get("data").get("train_file")
    )

    all_mappings = {
        "toks": pretrained_matrix_mapping,
        "chrs": char_mapping,
        "lbls": label_mapping,
    }

    # Dumping pretrained matrix
    # =========================
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
            json.dump(pretrained_matrix_size, output_file)

    target_mapping_file = os.path.join(
        os.path.abspath(output_dir), "mappings.json"
    )
    logging.debug("Dumping mappings to disk: {}".format(target_mapping_file))
    with open(target_mapping_file, "w", encoding="UTF-8") as output_file:
        json.dump(all_mappings, output_file)

    torch.set_num_threads(options.get("training").get("num_global_workers"))

    logging.debug("Creating logger")
    target_tensorboard_path = os.path.join(output_dir, "tb")
    train_logger = TrainLogger(tensorboard_path=target_tensorboard_path)

    logging.info("Creating dataloaders")
    dataloader_train, len_train, _ = create_dataloader(
        mappings=all_mappings,
        options=options,
        instance_json_file=options.get("data").get("train_file"),
        test=False,
        working_dir=output_dir,
    )

    dataloader_dev, len_dev, _ = create_dataloader(
        mappings=all_mappings,
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
        mappings=all_mappings,
        embedding_root_dir=os.path.join(output_dir, "embeddings"),
    )

    constraints = allowed_transitions(
        options.get("data").get("format"),
        {v: k for k, v in all_mappings.get("lbls").items()},
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
        mappings=all_mappings,
        nb_layers=options.get("network_structure")
        .get("lstm")
        .get("nb_layers"),
        num_labels=len(all_mappings.get("lbls")),
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
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
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
        lr_scheduler = None

    # SCHEDULER
    # =========
    max_step = int(
        len_train
        // (
            options.get("training").get("train_batch_size")
            * options.get("training").get("accumulation_steps")
        )
        * options.get("training").get("num_epochs")
    )

    warmup_scheduler = None

    eval_every_n_steps = int(
        len_train
        // (
            options.get("training").get("train_batch_size")
            * options.get("training").get("accumulation_steps")
        )
        * options.get("training").get("eval_every_%")
    )

    if eval_every_n_steps == 0:
        eval_every_n_steps = 1

    if options.get("training").get("warmup_scheduler").get("use"):
        num_warmup_step = max_step * options.get("training").get(
            "warmup_scheduler"
        ).get("%_warmup_steps")
        logging.info(
            "Using linear schedule with warmup (warmup-steps={}, total-steps={})".format(
                round(num_warmup_step), max_step
            )
        )
        warmup_scheduler = get_linear_schedule_with_warmup(
            optimizer, round(num_warmup_step), max_step
        )

    if options.get("training").get("cuda"):
        logging.info("Switching to cuda")
        model.cuda()

    if options.get("training").get("fp16"):
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
        batch_size=options.get("training").get("train_batch_size"),
        clip_grad_norm=options.get("training").get("clip_grad_norm"),
        cuda=options.get("training").get("cuda"),
        dataloader_train=dataloader_train,
        dataloader_dev=dataloader_dev,
        eval_function=eval_ner,
        eval_every_n_steps=eval_every_n_steps,
        fp16=options.get("training").get("fp16"),
        len_dataset_train=len_train,
        len_dataset_dev=len_dev,
        log_to_stdout_every_n_step=1,
        lr_scheduler=lr_scheduler,
        max_steps=max_step,
        model=model,
        optimizer=optimizer,
        train_logger=train_logger,
        warmup_scheduler=warmup_scheduler,
        working_dir=os.path.abspath(output_dir),
    )

    trainer.perform_training()
