import configparser
import logging
import os

import pkg_resources

from .data.reader import TestData
from .helpers.config import extract_params
from .tools import ensure_dir, log_message
from .nn.test import test_model


def apply_model(model_path, input_file, working_dir, timestamp):

    # Creating working directory
    current_working_directory = os.path.join(working_dir, "yaset-apply-{}".format(timestamp))
    ensure_dir(current_working_directory)

    # Setting up a log file and adding a new handler to the logger
    log_file = os.path.join(current_working_directory, "{}.log".format(
        "yaset-apply-{}".format(timestamp)
    ))

    # Setting up logger
    log_format = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger('')

    fh = logging.FileHandler(log_file, encoding="UTF-8")
    fh.setFormatter(log_format)
    log.addHandler(fh)

    log_message("BEGIN - LOADING AND CHECKING DATA FILES")

    data = TestData(input_file, working_dir=current_working_directory, train_model_path=model_path)

    data.check_input_file()

    log_message("END - LOADING AND CHECKING DATA FILES")

    log_message("BEGIN - CREATING TFRECORDS FILES")

    target_tfrecords_file_path = os.path.join(os.path.abspath(current_working_directory), "data.tfrecords")

    data.convert_to_tfrecords(input_file, target_tfrecords_file_path)

    log_message("END - CREATING TFRECORDS FILES")

    log_message("BEGIN - LOADING MODEL PARAMETERS")

    # Load config file used during training
    parsed_configuration = configparser.ConfigParser()
    parsed_configuration.read(os.path.join(model_path, "config.ini"))

    # Computing parameter description file paths
    general_param_desc_file = pkg_resources.resource_filename('yaset', 'desc/GENERAL_PARAMS_DESC.json')
    training_param_desc_file = pkg_resources.resource_filename('yaset', 'desc/TRAINING_PARAMS_DESC.json')
    bilstmcharcrf_param_desc_file = pkg_resources.resource_filename('yaset', 'desc/BILSTMCHARCRF_PARAMS_DESC.json')

    # Extracting parameters from configuration file according to parameter description files

    general_params = extract_params(parsed_configuration["general"], general_param_desc_file)
    logging.info("General section: OK")

    training_params = extract_params(parsed_configuration["training"], training_param_desc_file)
    logging.info("Training section: OK")

    if training_params["model_type"] == "bilstm-char-crf":
        model_params = extract_params(parsed_configuration["bilstm-char-crf"], bilstmcharcrf_param_desc_file)
    else:
        raise Exception("The model type you specified does not exist: {}".format(training_params["model_type"]))

    if general_params["batch_mode"]:
        model_indexes = list(range(general_params["batch_iter"]))
        logging.info("Operating in batch mode")
    else:
        model_indexes = [0]
        logging.info("Operating in single model mode")

    log_message("END - LOADING MODEL PARAMETERS")

    for i in model_indexes:

        log_message("BEGIN - APPLYING MODEL #{:03d}".format(i + 1))

        test_model(current_working_directory, model_path, data, training_params, model_params, i + 1)

        log_message("END - APPLYING MODEL #{:03d}".format(i + 1))
