import json
import os


def get_feature_columns(feature_list_str):
    """
    Parse the feature column parameter
    :param feature_list_str: feature column parameter value (string)
    :return: column list
    """

    parts = feature_list_str.split(',')
    parts = [int(item.strip(" ")) for item in parts]

    return parts


def extract_params(config_section, param_desc_file):
    """
    Extract params from a particular section in the config file.
    :param config_section: configuration section from where parameters will be extracted
    :param param_desc_file: parameter description
    :return: dictionary of parameters
    """

    param_desc = json.load(open(os.path.abspath(param_desc_file), "r", encoding="UTF-8"))

    parameters = _params_recur(config_section, param_desc)

    return parameters


def _params_recur(config_section, param_desc):
    """
    Recursive function for parameter extraction. Will iterate trough the parameter file.
    :param config_section: configuration section from where parameters will be extracted
    :param param_desc: parameter description
    :return: dictionary of parameters
    """

    all_params = dict()

    int_params = extract_int_values(config_section, param_desc.get("int_parameters"))
    for k, v in int_params.items():
        all_params[k] = v

    float_params = extract_float_values(config_section, param_desc.get("float_parameters"))
    for k, v in float_params.items():
        all_params[k] = v

    string_params = extract_string_values(config_section, param_desc.get("string_parameters"))
    for k, v in string_params.items():
        all_params[k] = v

    boolean_params = extract_boolean_values(config_section, param_desc.get("boolean_parameters"))
    for k, v in boolean_params.items():
        all_params[k] = v

    if param_desc.get("true_cond_parameters"):
        for p in param_desc.get("true_cond_parameters"):
            if config_section[p].lower() == "true":
                all_params[p] = True
                cond_params = _params_recur(config_section, param_desc["true_cond_parameters"][p])
                for k, v in cond_params.items():
                    all_params[k] = v
            elif config_section[p].lower() == "false":
                all_params[p] = False

    if param_desc.get("false_cond_parameters"):
        for p in param_desc.get("false_cond_parameters"):
            if config_section[p].lower() == "false":
                all_params[p] = False
                cond_params = _params_recur(config_section, param_desc["false_cond_parameters"][p])
                for k, v in cond_params.items():
                    all_params[k] = v
            elif config_section[p].lower() == "true":
                all_params[p] = True

    if param_desc.get("string_cond_parameters"):
        for p in param_desc.get("string_cond_parameters"):
            for value in param_desc["string_cond_parameters"][p]:
                if config_section.get(p) == value:
                    all_params[p] = value
                    cond_params = _params_recur(config_section, param_desc["string_cond_parameters"][p][value])
                    for k, v in cond_params.items():
                        all_params[k] = v

    return all_params


def extract_int_values(config_section, param_list):
    """
    Extract integer parameters
    :param config_section: configuration section from where parameters will be extracted
    :param param_list: list of integer parameters to extract
    :return: dict of parameters
    """

    params = dict()

    if param_list:
        for p in param_list:
            params[p] = int(config_section.get(p))

    return params


def extract_float_values(config_section, param_list):
    """
    Extract float parameters
    :param config_section: configuration section from where parameters will be extracted
    :param param_list: list of float parameters to extract
    :return: dict of parameters
    """

    params = dict()

    if param_list:
        for p in param_list:
            params[p] = float(config_section.get(p))

    return params


def extract_string_values(config_section, param_list):
    """
    Extract string parameters
    :param config_section: configuration section from where parameters will be extracted
    :param param_list: list of string parameters to extract
    :return: dict of parameters
    """

    params = dict()

    if param_list:
        for p in param_list:
            params[p] = str(config_section.get(p))

    return params


def extract_boolean_values(config_section, param_list):
    """
    Extract boolean parameters
    :param config_section: configuration section from where parameters will be extracted
    :param param_list: list of boolean parameters to extract
    :return: dict of parameters
    """

    params = dict()

    if param_list:
        for p in param_list:
            if config_section.get(p).lower() == "true":
                params[p] = True
            elif config_section.get(p).lower() == "false":
                params[p] = False
            else:
                raise Exception("The value for the attribute {} should be a boolean (true or false), got {}".format(
                    p, config_section.get(p)
                ))

    return params
