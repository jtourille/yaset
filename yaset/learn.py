import importlib
import logging
import os

import pkg_resources

from .data.reader import TrainData
from .helpers.config import extract_params
from .nn.train import train_model
from .tools import ensure_dir, log_message


def learn_model(parsed_configuration, timestamp):

    # ---------------------------------------------------------------
    # PARAMETER LOADING

    # Computing parameter description file paths
    training_param_desc_file = pkg_resources.resource_filename('yaset', 'desc/TRAINING_PARAMS_DESC.json')
    data_param_desc_file = pkg_resources.resource_filename('yaset', 'desc/DATA_PARAMS_DESC.json')
    bilstmcharcrf_param_desc_file = pkg_resources.resource_filename('yaset', 'desc/BILSTMCHARCRF_PARAMS_DESC.json')

    # Extracting parameters from configuration file according to parameter description files
    data_params = extract_params(parsed_configuration["data"], data_param_desc_file)
    training_params = extract_params(parsed_configuration["training"], training_param_desc_file)

    # Checking if the nn model is implemented in yaset
    if training_params["model_type"] == "bilstm-char-crf":
        model_params = extract_params(parsed_configuration["bilstm-char-crf"], bilstmcharcrf_param_desc_file)
    else:
        raise Exception("The model type you specified does not exist: {}".format(training_params["model_type"]))

    # Checking if the working directory specified in the configuration exists
    if not os.path.isdir(os.path.abspath(data_params.get("working_dir"))):
        raise NotADirectoryError("The working directory you specified does not exist: {}".format(
            os.path.abspath(data_params.get("working_dir"))
        ))

    # ---------------------------------------------------------------
    # WORKING DIRECTORY SETUP

    # Creating the current working directory based on the top working directory
    current_working_directory = os.path.join(
        os.path.abspath(data_params.get("working_dir")),
        "yaset-learn-{}".format(timestamp)
    )
    ensure_dir(current_working_directory)

    # ---------------------------------------------------------------
    # LOGGING

    log_format = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger('')

    # Setting up a log file and adding a new handler to the logger
    log_file = os.path.join(current_working_directory, "{}.log".format(
        "yaset-learn-{}".format(timestamp)
    ))

    fh = logging.FileHandler(log_file, encoding="UTF-8")
    fh.setFormatter(log_format)
    log.addHandler(fh)

    # -----------------------------------------------------------
    # FEATURES - NOT FULLY IMPLEMENTED YET

    feature_columns = list()
    #
    # if parsed_configuration.getboolean("data", "use_features"):
    #     feature_columns = parse_feature_columns(parsed_configuration["data"]["feature_columns"])

    # -----------------------------------------------------------
    # LOADING AND CHECKING DATA FILES

    log_message("BEGIN - LOADING AND CHECKING DATA FILES")

    # Creating data object
    data = TrainData(working_dir=current_working_directory, data_params=data_params)

    # Checking file format
    data.check_input_files()

    log_message("END - LOADING AND CHECKING DATA FILES")

    # -----------------------------------------------------------
    # EMBEDDING LOADING AND PROCESSING

    log_message("BEGIN - EMBEDDING LOADING AND PREPROCESSING")

    embedding_model_type = data_params.get("embedding_model_type")

    if embedding_model_type is not "random":
        # Case where the model type is not random

        logging.info("Model type: {}".format(embedding_model_type))

        # Checking if the embedding model path does exist
        embedding_file_path = os.path.abspath(data_params.get("embedding_model_path"))

        if not os.path.isfile(embedding_file_path):
            raise FileNotFoundError("The embedding file you specified doesn't exist: {}".format(
                embedding_file_path
            ))

        logging.info("File path: {}".format(embedding_file_path))

        embedding_oov_strategy = data_params.get("embedding_oov_strategy")
        embedding_oov_map_token_id = None
        embedding_oov_replace_rate = None

        if embedding_oov_strategy == "map":
            embedding_oov_map_token_id = data_params.get("embedding_oov_map_token_id")

        elif embedding_oov_strategy == "replace":
            embedding_oov_replace_rate = data_params.get("embedding_oov_replace_rate")

        elif embedding_oov_strategy == "none":
            pass

        else:
            raise Exception("The OOV strategy you specified is not recognized: {}".format(embedding_oov_strategy))

        # Dynamic loading of embedding module. Allow to write custom modules for specific model formats.
        logging.debug("Creating embedding object")
        embedding_module = importlib.import_module("yaset.embed.{}".format(embedding_model_type))
        embedding_class = getattr(embedding_module, "{}Embeddings".format(embedding_model_type.title()))
        embedding_object = embedding_class(embedding_file_path, embedding_oov_strategy, embedding_oov_map_token_id)

        # Loading embedding matrix into embedding object
        logging.info("Loading matrix")
        embedding_object.load_embedding()

        if embedding_oov_strategy == "replace":
            logging.info("Building unknown token vector")
            _ = embedding_object.build_unknown_token()

        elif embedding_oov_strategy == "map":
            logging.info("Unknown token vector already exists, skipping building new one (id={})".format(
                embedding_oov_map_token_id
            ))

        elif embedding_oov_strategy == "none":
            logging.info("No unknown token vector is build")

    else:
        # Random embedding will be supported in a later release
        raise Exception("Random embeddings are not supported yet")

    log_message("END - EMBEDDING LOADING AND PREPROCESSING")

    log_message("BEGIN - CREATING TFRECORDS FILES")

    data.create_tfrecords_files(embedding_object, oov_strategy=embedding_oov_strategy,
                                unk_token_rate=embedding_oov_replace_rate)

    log_message("END - CREATING TFRECORDS FILES")

    log_message("BEGIN - LEARNING MODEL")

    logging.debug("Current training parameters")
    for k, v in training_params.items():
        logging.debug("* {} = {}".format(k, v))

    train_model(current_working_directory, embedding_object, data, training_params, model_params)

    log_message("END - LEARNING MODEL")

    return current_working_directory
