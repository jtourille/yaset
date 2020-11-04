import logging
import os
import shutil
import sys

import click
from yaset.single.train import train_single_model


@click.group()
@click.option("--debug", is_flag=True)
def cli(debug):
    log = logging.getLogger("")
    log.handlers = []
    log_format = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    if debug:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    # Adding a stdout handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(log_format)
    log.addHandler(ch)


@cli.command("TRAIN-SINGLE")
@click.option(
    "--config-file",
    help="Training configuration file",
    type=str,
    required=True,
)
@click.option(
    "--output-dir",
    help="Output directory where files will be written",
    type=str,
    required=True,
)
def train(config_file, output_dir):
    """Train a single model"""

    config_file = os.path.abspath(config_file)
    output_dir = os.path.abspath(output_dir)

    if not os.path.isfile(config_file):
        raise FileNotFoundError(
            "The configuration file does not exist: {}".format(config_file)
        )

    if os.path.isdir(output_dir):
        click.confirm(
            "The output directory already exists. Do you want to overwrite?",
            abort=True,
        )
        click.echo("Overwriting output directory: {}".format(output_dir))
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    log_file = os.path.join(os.path.abspath(output_dir), "training.log")
    log_format = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("")

    fh = logging.FileHandler(log_file, encoding="UTF-8")
    fh.setFormatter(log_format)
    log.addHandler(fh)

    train_single_model(option_file=config_file, output_dir=output_dir)
