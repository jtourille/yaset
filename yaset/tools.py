import os
import logging


def ensure_dir(directory):

    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        # Raising any errors except concurrent access
        if e.errno != 17:
            raise


def remove_abs(path):

    if os.path.isabs(path):
        return path.lstrip("/")
    else:
        return path


def get_other_extension(filename, target_extension):

    basename, extension = os.path.splitext(filename)

    return "{0}.{1}".format(basename, target_extension)


def log_message(message, symbol="="):

    logging.info("{} {} {}".format(
        symbol * 10,
        message,
        symbol * (70 - len(message) - 2 - 10)
    ))
