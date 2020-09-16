import json
import logging
import os

import _jsonnet
import torch
import torch.nn as nn
from yaset.ensemble.inference import NERModel
from yaset.ensemble.train import (
    load_mappings,
    load_options,
    SENTENCE_SIZE_MAPPING,
)
from yaset.nn.ensemble import EnsembleWithAttention
from yaset.utils.config import replace_auto
from yaset.utils.load import load_model
from yaset.utils.logging import TrainLogger


def load_model_ensemble(model_dir: str = None, cuda: bool = None):
    """
    Load NER ensemble model

    Args:
        model_dir (str): NER ensemble model directory
        cuda (bool): activate CUDA

    Returns:
        NERModel: NER ensemble model
    """

    model, model_options, model_mappings = load_model_ensemble_helper(
        model_dir=model_dir
    )

    option_file = os.path.join(os.path.abspath(model_dir), "options.jsonnet")
    options_json = _jsonnet.evaluate_file(option_file)
    options = json.loads(options_json)
    replace_auto(options=options)

    if cuda:
        logging.info("Switching to cuda")
        model.cuda()

    ner_model = NERModel(
        model_mappings=model_mappings,
        model=model,
        model_options=model_options,
        sentence_size_mapping=SENTENCE_SIZE_MAPPING,
    )

    return ner_model


def load_model_ensemble_helper(model_dir: str = None):
    """
    Helper function for NER ensemble loading

    Args:
        model_dir (str): NER ensemble model directory

    Returns:
        model: NER ensemble model

    """

    option_file = os.path.join(os.path.abspath(model_dir), "options.jsonnet")
    logging_file = os.path.join(os.path.abspath(model_dir), "train_log.pkl")
    parameter_dir = os.path.join(os.path.abspath(model_dir), "models")
    tensorboard_dir = os.path.join(os.path.abspath(model_dir), "tb")

    model_1 = None
    model_2 = None
    model_3 = None
    model_4 = None
    model_5 = None

    if os.path.isdir(os.path.join(model_dir, "input_models", "model-1")):
        model_1 = os.path.join(model_dir, "input_models", "model-1")

    if os.path.isdir(os.path.join(model_dir, "input_models", "model-2")):
        model_2 = os.path.join(model_dir, "input_models", "model-2")

    if os.path.isdir(os.path.join(model_dir, "input_models", "model-3")):
        model_3 = os.path.join(model_dir, "input_models", "model-3")

    if os.path.isdir(os.path.join(model_dir, "input_models", "model-4")):
        model_4 = os.path.join(model_dir, "input_models", "model-4")

    if os.path.isdir(os.path.join(model_dir, "input_models", "model-5")):
        model_5 = os.path.join(model_dir, "input_models", "model-5")

    options_json = _jsonnet.evaluate_file(option_file)
    options = json.loads(options_json)
    replace_auto(options=options)

    model_mappings = load_mappings(
        model_1=model_1,
        model_2=model_2,
        model_3=model_3,
        model_4=model_4,
        model_5=model_5,
    )

    model_options = load_options(
        model_1=model_1,
        model_2=model_2,
        model_3=model_3,
        model_4=model_4,
        model_5=model_5,
    )

    train_logger = TrainLogger(tensorboard_path=tensorboard_dir)
    train_logger.load_json_file(filepath=logging_file)

    best_idx, _ = train_logger.get_best_iteration()

    model_file = os.path.join(parameter_dir, "model-{}.pth".format(best_idx))

    logging.info("Loading models")
    models = nn.ModuleDict()

    logging.info("* {}".format("Model 1"))
    models["model_1"] = load_model(model_dir=model_1)

    logging.info("* {}".format("Model 2"))
    models["model_2"] = load_model(model_dir=model_2)

    if model_3:
        logging.info("* {}".format("Model 3"))
        models["model_3"] = load_model(model_dir=model_3)

    if model_4:
        logging.info("* {}".format("Model 4"))
        models["model_4"] = load_model(model_dir=model_4)

    if model_5:
        logging.info("* {}".format("Model 5"))
        models["model_5"] = load_model(model_dir=model_5)

    reference_id = "model_1"

    # constraints = allowed_transitions(options.get("data").get("format"),
    #                                   {v: k for k, v in model_mappings[reference_id]["ner_labels"].items()})

    # if options.get("type") == "label-stacking":
    #     model = LabelStacking(
    #         constraints=constraints,
    #         ffnn_hidden_layer_use=options.get("network_structure").get("ffnn").get("use"),
    #         ffnn_hidden_layer_size=options.get("network_structure").get("ffnn").get("hidden_layer_size"),
    #         ffnn_activation_function=options.get("network_structure").get("ffnn").get("activation_function"),
    #         ffnn_input_dropout_rate=options.get("network_structure").get("ffnn").get("input_dropout_rate"),
    #         label_embedding_size=options.get("network_structure").get("label_embedding_size"),
    #         lstm_hidden_size=options.get("network_structure").get("lstm_hidden_size"),
    #         lstm_input_dropout_rate=options.get("network_structure").get("lstm_input_dropout_rate"),
    #         lstm_layer_dropout_rate=options.get("network_structure").get("lstm_layer_dropout_rate"),
    #         use_highway=options.get("network_structure").get("use_highway"),
    #         mappings=model_mappings[reference_id],
    #         models=models,
    #         num_labels=len(model_mappings[reference_id]["ner_labels"]),
    #         nb_layers=options.get("network_structure").get("nb_layers"),
    #         reference_id=reference_id,
    #         num_sentence_bins=len(SENTENCE_SIZE_MAPPING),
    #         sentence_embedding_size=options.get("network_structure").get("sentence_embedding_size")
    #     )
    # elif options.get("type") == "ensemble-lstm":
    #     model = EnsembleLSTM(
    #         constraints=constraints,
    #         ffnn_hidden_layer_use=options.get("network_structure").get("ffnn").get("use"),
    #         ffnn_hidden_layer_size=options.get("network_structure").get("ffnn").get("hidden_layer_size"),
    #         ffnn_activation_function=options.get("network_structure").get("ffnn").get("activation_function"),
    #         ffnn_input_dropout_rate=options.get("network_structure").get("ffnn").get("input_dropout_rate"),
    #         lstm_hidden_size=options.get("network_structure").get("lstm_hidden_size"),
    #         lstm_input_dropout_rate=options.get("network_structure").get("lstm_input_dropout_rate"),
    #         lstm_layer_dropout_rate=options.get("network_structure").get("lstm_layer_dropout_rate"),
    #         use_highway=options.get("network_structure").get("use_highway"),
    #         mappings=model_mappings[reference_id],
    #         models=models,
    #         num_labels=len(model_mappings[reference_id]["ner_labels"]),
    #         nb_layers=options.get("network_structure").get("nb_layers"),
    #         reference_id=reference_id,
    #         num_sentence_bins=len(SENTENCE_SIZE_MAPPING),
    #         sentence_embedding_size=options.get("network_structure").get("sentence_embedding_size")
    #     )
    if options.get("type") == "ensemble-with-attention":
        model = EnsembleWithAttention(
            attention_final_hidden_size=options.get("network_structure").get(
                "attention_final_hidden_size"
            ),
            attention_final_dropout_rate=options.get("network_structure").get(
                "attention_final_dropout_rate"
            ),
            attention_mixin_hidden_size=options.get("network_structure").get(
                "attention_mixin_hidden_size"
            ),
            attention_mixin_dropout_rate=options.get("network_structure").get(
                "attention_mixin_dropout_rate"
            ),
            mappings=model_mappings[reference_id],
            models=models,
            num_labels=len(model_mappings[reference_id]["ner_labels"]),
            reference_id=reference_id,
        )

    else:
        raise Exception("The model does not exist")

    logging.debug("Loading weights")
    model.load_state_dict(torch.load(model_file, map_location="cpu"))

    return model, model_options, model_mappings
