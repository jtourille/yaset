import logging
import re

from .tools.conll import load_sentences
from .utils.load import load_model
from .utils.training import compute_steps


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def apply_model(model_dir: str = None,
                input_file: str = None,
                output_file: str = None,
                batch_size: int = 128,
                cuda: bool = False):

    sentences = load_sentences(input_file=input_file)
    model = load_model(model_dir=model_dir, cuda=cuda)

    logging.info("Starting predicting")
    labels = list()

    processed = 0
    steps = compute_steps(len(sentences), step=0.05)

    for batch in chunks(sentences, batch_size):
        labels.extend(model(batch, cuda))
        processed += len(batch)

        if processed >= steps[0] or processed == len(sentences):
            logging.info("Processed={} ({:6.2f})".format(processed,
                                                         (processed/len(sentences) * 100)))

    labels_flatten = [i for seq in labels for i in seq]
    labels_index = 0

    with open(output_file, "w", encoding="UTF-8") as o_file:
        with open(input_file, "r", encoding="UTF-8") as i_file:
            for line in i_file:
                if re.match("^$", line):
                    o_file.write(line)
                    continue

                o_line = line.rstrip("\n")
                o_line = "{}\t{}\n".format(o_line, labels_flatten[labels_index])

                o_file.write(o_line)

                labels_index += 1
