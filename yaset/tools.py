import os
import logging


def ensure_dir(directory):
    """
    Create a directory
    :param directory: directory to create
    :return: nothing
    """

    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        # Raising any errors except concurrent access
        if e.errno != 17:
            raise


def remove_abs(path):
    """
    Remove leading '/' from absolute paths
    :param path: input absolute path
    :return: transformed input absolute path
    """

    if os.path.isabs(path):
        return path.lstrip("/")
    else:
        return path


def get_other_extension(filename, target_extension):
    """
    Get anther extension of a file
    :param filename: input file path
    :param target_extension: new extension
    :return: transformed file path
    """

    basename, extension = os.path.splitext(filename)

    return "{0}.{1}".format(basename, target_extension)


def log_message(message, symbol="="):
    """
    Log a message
    :param message: input message
    :param symbol: separator to use
    :return: nothing
    """

    logging.info("{} {} {}".format(
        symbol * 5,
        message,
        symbol * (70 - len(message) - 2 - 5)
    ))
