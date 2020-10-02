import os
import shutil

from yaset.utils.path import ensure_dir


def copy_embedding_models(
    embeddings_options: dict = None, output_dir: str = None
):
    """
    Copy pretrained embeddings specified in configuration file to model directory

    :param embeddings_options: configuration file portion related to embeddings
    :type embeddings_options: dict

    :param output_dir: directory where files will be copied
    :type output_dir: str

    :return: None
    """

    if embeddings_options.get("elmo").get("use"):
        target_subdir = os.path.join(os.path.abspath(output_dir), "elmo")
        ensure_dir(target_subdir)

        target_weight = os.path.join(
            target_subdir,
            os.path.basename(
                embeddings_options.get("elmo").get("weight_path")
            ),
        )
        target_options = os.path.join(
            target_subdir,
            os.path.basename(
                embeddings_options.get("elmo").get("options_path")
            ),
        )

        shutil.copy(
            embeddings_options.get("elmo").get("weight_path"), target_weight
        )
        shutil.copy(
            embeddings_options.get("elmo").get("options_path"), target_options
        )

    if embeddings_options.get("bert").get("use"):
        target_subdir = os.path.join(os.path.abspath(output_dir), "bert")
        target_vocab_subdir = os.path.join(
            os.path.abspath(target_subdir), "vocab"
        )

        ensure_dir(target_subdir)
        ensure_dir(target_vocab_subdir)

        if embeddings_options.get("bert").get("type") == "pytorch":
            target_model_file = os.path.join(
                target_subdir,
                os.path.basename(
                    embeddings_options.get("bert").get("model_file")
                ),
            )
            target_model_vocab = os.path.join(target_vocab_subdir, "vocab.txt")
            target_model_config_file = os.path.join(
                target_subdir,
                os.path.basename(
                    embeddings_options.get("bert").get("config_file")
                ),
            )

            shutil.copy(
                embeddings_options.get("bert").get("model_file"),
                target_model_file,
            )
            shutil.copy(
                embeddings_options.get("bert").get("vocab_file"),
                target_model_vocab,
            )
            shutil.copy(
                embeddings_options.get("bert").get("config_file"),
                target_model_config_file,
            )
        else:
            target_model_dir = os.path.join(
                target_subdir,
                os.path.basename(
                    embeddings_options.get("bert").get("model_file")
                ),
            )
            target_model_vocab = os.path.join(
                target_subdir,
                os.path.basename(
                    embeddings_options.get("bert").get("vocab_file")
                ),
            )

            shutil.copytree(
                embeddings_options.get("bert").get("model_file"),
                target_model_dir,
            )
            shutil.copy(
                embeddings_options.get("bert").get("vocab_file"),
                target_model_vocab,
            )

    if embeddings_options.get("pretrained").get("use"):
        target_subdir = os.path.join(os.path.abspath(output_dir), "pretrained")
        ensure_dir(target_subdir)

        target_model = os.path.join(
            target_subdir,
            os.path.basename(
                embeddings_options.get("pretrained").get("model_path")
            ),
        )

        shutil.copy(
            embeddings_options.get("pretrained").get("model_path"),
            target_model,
        )
