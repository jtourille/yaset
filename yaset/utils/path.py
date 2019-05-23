import os


def ensure_dir(directory: str) -> None:
    """
    Creates a directory

    Args:
        directory (str): path to create

    Returns:
        None
    """

    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        # Raising any errors except concurrent access
        if e.errno != 17:
            raise
