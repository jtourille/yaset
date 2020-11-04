import logging
import os
import shutil
import sys

import click
from yaset.single.apply import apply_model
from yaset.single.train import train_single_model


@click.group()
@click.option("--debug", is_flag=True)
@click.pass_context
def cli(ctx, debug):
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

    ctx.ensure_object(dict)
    ctx.obj["DEBUG"] = debug


@cli.command("APPLY-SINGLE")
@click.option(
    "--model-dir",
    help="Directory where model is stored",
    type=str,
    required=True,
)
@click.option(
    "--output-dir",
    help="Output directory where files will be written",
    type=str,
    required=True,
)
@click.option(
    "--input-file",
    help="Fil where instances are stored",
    type=str,
    required=True,
)
@click.option("--cuda", is_flag=True)
@click.option(
    "--n-jobs", help="Number of CPU processes to use", type=int, default=1
)
@click.pass_context
def apply(ctx, model_dir, output_dir, input_file, cuda, n_jobs):
    """Apply a single model"""

    model_dir = os.path.abspath(model_dir)
    output_dir = os.path.abspath(output_dir)
    input_file = os.path.abspath(input_file)

    if not os.path.isdir(model_dir):
        raise NotADirectoryError(
            "The model directory does not exist: {}".format(model_dir)
        )

    if not os.path.isfile(input_file):
        raise FileNotFoundError(
            "The input file does not exist: {}".format(input_file)
        )

    if os.path.isdir(output_dir):
        click.confirm(
            "The output directory already exists. Do you want to overwrite?",
            abort=True,
        )
        click.echo("Overwriting output directory: {}".format(output_dir))
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    log_file = os.path.join(os.path.abspath(output_dir), "inference.log")
    log_format = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("")

    fh = logging.FileHandler(log_file, encoding="UTF-8")
    fh.setFormatter(log_format)
    log.addHandler(fh)

    output_file = os.path.join(output_dir, "output.conll")

    apply_model(
        model_dir=model_dir,
        input_file=input_file,
        output_file=output_file,
        cuda=cuda,
        n_jobs=n_jobs,
        debug=ctx.obj["DEBUG"],
    )


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
