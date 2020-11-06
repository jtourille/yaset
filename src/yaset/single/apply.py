import logging
import re

import torch
from yaset.tools.conll import load_sentences
from yaset.utils.load import load_model_single


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def apply_model(
    model_dir: str = None,
    input_file: str = None,
    output_file: str = None,
    batch_size: int = 128,
    cuda: bool = False,
    n_jobs: int = None,
    debug: bool = False,
):

    torch.set_num_threads(n_jobs)

    sentences = load_sentences(input_file=input_file, debug=debug)
    model = load_model_single(model_dir=model_dir, cuda=cuda)

    logging.info("Starting predicting")
    labels = list()

    processed = 0

    for batch in chunks(sentences, batch_size):
        labels.extend(model(batch, cuda))
        processed += len(batch)

        logging.info(
            "Processed={} ({:6.2f})".format(
                processed, (processed / len(sentences) * 100)
            )
        )

    labels_flatten = [i for seq in labels for i in seq]
    labels_index = 0

    with open(output_file, "w", encoding="UTF-8") as o_file:
        with open(input_file, "r", encoding="UTF-8") as i_file:
            for line in i_file:
                if re.match("^$", line):
                    o_file.write(line)
                    continue

                o_line = line.rstrip("\n")
                if debug and labels_index >= len(labels_flatten):
                    o_line = "{}\n".format(o_line)

                else:
                    o_line = "{}\t{}\n".format(
                        o_line, labels_flatten[labels_index]
                    )

                o_file.write(o_line)

                labels_index += 1
