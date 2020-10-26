def replace_auto(options: dict = None) -> None:
    """
    Replace the keyword 'auto' by '-1' in configuration files

    :param options: configuration parameters
    :type options: dict

    :return: None
    """
    for key, value in options.items():
        if isinstance(value, dict):
            replace_auto(options=value)
        else:
            if value == "auto":
                options[key] = -1
