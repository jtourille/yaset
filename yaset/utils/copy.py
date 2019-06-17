import os
import shutil

from .path import ensure_dir


def copy_embedding_models(embeddings_options: dict = None,
                          output_dir: str = None):

    if embeddings_options.get("elmo").get("use"):
        target_subdir = os.path.join(os.path.abspath(output_dir), "elmo")
        ensure_dir(target_subdir)

        target_weight = os.path.join(target_subdir,
                                     os.path.basename(embeddings_options.get("elmo").get("weight_path")))
        target_options = os.path.join(target_subdir,
                                      os.path.basename(embeddings_options.get("elmo").get("options_path")))

        shutil.copy(embeddings_options.get("elmo").get("weight_path"), target_weight)
        shutil.copy(embeddings_options.get("elmo").get("options_path"), target_options)

    if embeddings_options.get("bert").get("use"):
        target_subdir = os.path.join(os.path.abspath(output_dir), "bert")
        shutil.copytree(os.path.abspath(embeddings_options.get("bert").get("model_root_dir")), target_subdir)
