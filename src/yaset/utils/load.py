import json
import logging
import os

import _jsonnet
import torch
from yaset.nn.crf import allowed_transitions
from yaset.nn.embedding import Embedder
from yaset.nn.lstmcrf import AugmentedLSTMCRF
from yaset.single.inference import NERModel
from yaset.utils.config import replace_auto
from yaset.utils.logging import TrainLogger


def load_model_single(model_dir: str = None, cuda: bool = None):

    mapping_file = os.path.join(os.path.abspath(model_dir), "mappings.json")
    option_file = os.path.join(os.path.abspath(model_dir), "options.jsonnet")

    with open(mapping_file, "r", encoding="UTF-8") as input_file:
        mappings = json.load(input_file)

    options_json = _jsonnet.evaluate_file(option_file)
    options = json.loads(options_json)
    replace_auto(options=options)

    model = load_model(model_dir=model_dir)

    if cuda:
        logging.info("Switching to cuda")
        model.cuda()

    model.eval()

    ner_model = NERModel(
        mappings=mappings, model=model, options=options, model_dir=model_dir
    )

    return ner_model


def load_model(model_dir: str = None):
    """
    Load a single NER model

    Args:
        model_dir (str): NER model directory

    Returns:
        NER model
    """
    mapping_file = os.path.join(os.path.abspath(model_dir), "mappings.json")
    option_file = os.path.join(os.path.abspath(model_dir), "options.jsonnet")
    logging_file = os.path.join(os.path.abspath(model_dir), "train_log.pkl")
    parameter_dir = os.path.join(os.path.abspath(model_dir), "models")
    tensorboard_dir = os.path.join(os.path.abspath(model_dir), "tb")
    pretrained_matrix_size_file = os.path.join(
        os.path.abspath(model_dir), "pretrained_matrix_size.json"
    )

    pretrained_matrix = None
    pretrained_matrix_size = None
    if os.path.isfile(pretrained_matrix_size_file):
        with open(
            pretrained_matrix_size_file, "r", encoding="UTF-8"
        ) as output_file:
            pretrained_matrix_size = json.load(output_file)

    options_json = _jsonnet.evaluate_file(option_file)
    options = json.loads(options_json)
    replace_auto(options=options)

    with open(mapping_file, "r", encoding="UTF-8") as input_file:
        mappings = json.load(input_file)

    train_logger = TrainLogger(tensorboard_path=tensorboard_dir)
    train_logger.load_json_file(filepath=logging_file)

    best_idx, _ = train_logger.get_best_step(criterion="f1_score")

    model_file = os.path.join(parameter_dir, "model-{}.pth".format(best_idx))

    logging.info("Building model")
    embedder = Embedder(
        embeddings_options=options.get("embeddings"),
        pretrained_matrix=pretrained_matrix,
        pretrained_matrix_size=pretrained_matrix_size,
        mappings=mappings,
        embedding_root_dir=os.path.join(model_dir, "embeddings"),
    )

    constraints = allowed_transitions(
        options.get("data").get("format"),
        {v: k for k, v in mappings.get("lbls").items()},
    )

    # model = AugmentedLSTMCRF(
    #     embedder=embedder,
    #     constraints=constraints,
    #     ffnn_hidden_layer_use=options.get("network_structure")
    #     .get("ffnn")
    #     .get("use"),
    #     ffnn_hidden_layer_size=options.get("network_structure")
    #     .get("ffnn")
    #     .get("hidden_layer_size"),
    #     ffnn_activation_function=options.get("network_structure")
    #     .get("ffnn")
    #     .get("activation_function"),
    #     ffnn_input_dropout_rate=options.get("network_structure")
    #     .get("ffnn")
    #     .get("input_dropout_rate"),
    #     input_size=embedder.embedding_size,
    #     lstm_input_dropout_rate=options.get("network_structure").get(
    #         "lstm_input_dropout_rate"
    #     ),
    #     lstm_hidden_size=options.get("network_structure").get("hidden_size"),
    #     lstm_layer_dropout_rate=options.get("network_structure").get(
    #         "lstm_layer_dropout_rate"
    #     ),
    #     mappings=mappings,
    #     nb_layers=options.get("network_structure").get("nb_layers"),
    #     num_labels=len(mappings["ner_labels"]),
    #     use_highway=options.get("network_structure").get("use_highway"),
    # )

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
        embedding_input_size=embedder.embedding_size,
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
        lstm_nb_layers=options.get("network_structure")
        .get("lstm")
        .get("nb_layers"),
        num_labels=len(mappings.get("lbls")),
        lstm_use_highway=options.get("network_structure")
        .get("lstm")
        .get("highway"),
    )

    logging.debug("Loading weights")
    model.load_state_dict(torch.load(model_file, map_location="cpu"))

    return model
