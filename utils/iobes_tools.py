import argparse
import logging
import os
import re
import time
from datetime import timedelta


def iob_to_iobes(input_file, output_file):
    """
    Convert conll2003 file by changing tagging scheme from iob to iobes
    :param input_file: input conll2003 file (must be tab-separated)
    :param output_file: output conll2003 file
    :return: nothing
    """

    # Will contains all sequences from the input file
    all_sequences = list()

    with open(os.path.abspath(input_file), "r", encoding="UTF-8") as input_file:

        current_sequence = list()

        # Gathering sequences
        for i, line in enumerate(input_file, start=1):
            if re.match("^$", line):
                if len(current_sequence) > 0:
                    all_sequences.append(current_sequence.copy())
                    current_sequence.clear()

                continue

            # Line with tokens are split and stored
            parts = line.rstrip("\n").split("\t")  # Splitting line
            current_sequence.append(parts)

        # End of file, adding information about the last sequence if necessary
        if len(current_sequence) > 0:
            all_sequences.append(current_sequence)

    with open(os.path.abspath(output_file), "w", encoding="UTF-8") as output_file:
        # Looping through sentences and converting scheme
        for sequence in all_sequences:

            # Converting tagging scheme
            payload = _iob_to_iobes_seq(sequence)

            # Writing to file
            for item in payload:
                output_file.write("{}\n".format("\t".join(item)))

            # Separating sequences with blank lines
            output_file.write("\n")


def _iob_to_iobes_seq(sequence):
    """
    Convert tagging scheme from iob to iobes
    :param sequence: sequence to convert
    :return: converted sequence
    """

    # Fetching current labels
    source_labels = [item[-1] for item in sequence]
    target_labels = list()

    for i, label in enumerate(source_labels):

        # Fetching tag and label for current token
        if label.startswith("O"):
            current_tag = "O"
            current_cat = None
        else:
            if label.count('-') == 1:
                current_tag, current_cat = label.split("-")
            else:
                parts = label.split("-")
                current_tag = parts[0]
                current_cat = "-".join(parts[1:])

        next_tag, next_cat = None, None

        if i < len(source_labels) - 1:
            if source_labels[i+1].startswith("O"):
                next_tag = "O"
            else:
                if source_labels[i+1].count('-') == 1:
                    next_tag, next_cat = source_labels[i+1].split("-")
                else:
                    parts = source_labels[i+1].split("-")
                    next_tag = parts[0]
                    next_cat = "-".join(parts[1:])

        # Case where current tag is 'O', no change
        if current_tag == "O":
            target_labels.append(label)

        # Case where current tag is 'B'
        elif current_tag == "B":

            # If next tag is 'O' or 'B', the token is a singleton
            if next_tag in ["O", "B"] or i == len(source_labels)-1:
                target_labels.append("{}-{}".format("S", current_cat))
            # In the third case the next tag if "I"
            else:
                target_labels.append(label)

        # Case where current tag is 'I'
        elif current_tag == "I":

            # If next tag is 'O' or 'B, or if this is the last token of the sentence
            if next_tag == "O" or next_tag == "B" or i == len(source_labels)-1:
                target_labels.append("{}-{}".format("E", current_cat))

            # Case where next tag is "I"
            else:
                target_labels.append(label)

    # Repacking sequence and returning
    final_bioes = list()

    for source_item, target_label in zip(sequence, target_labels):
        final_bioes.append(source_item[:-1] + [target_label])

    return final_bioes


def iobes_to_iob(input_file, output_file):
    """
    Convert yaset output file by changing tagging scheme from iobes to iob
    :param input_file: yaset output file (must be tab-separated)
    :param output_file: corrected yaset file
    :return: nothing
    """

    # Will contains all sequences from the input file
    all_sequences = list()

    with open(os.path.abspath(input_file), "r", encoding="UTF-8") as input_file:

        current_sequence = list()

        # Gathering sequences
        for i, line in enumerate(input_file, start=1):
            if re.match("^$", line):
                if len(current_sequence) > 0:
                    all_sequences.append(current_sequence.copy())
                    current_sequence.clear()

                continue

            # Line with tokens are split and stored
            parts = line.rstrip("\n").split("\t")  # Splitting line
            current_sequence.append(parts)

        # End of file, adding information about the last sequence if necessary
        if len(current_sequence) > 0:
            all_sequences.append(current_sequence)

    with open(os.path.abspath(output_file), "w", encoding="UTF-8") as output_file:
        # Looping through sentences and converting scheme
        for sequence in all_sequences:

            # Converting tagging scheme
            payload = _iobes_to_iob_seq(sequence)

            # Writing to file
            for item in payload:
                output_file.write("{}\n".format("\t".join(item)))

            # Separating sequences with blank lines
            output_file.write("\n")


def _iobes_to_iob_seq(sequence):
    """
    Convert tagging scheme from iobes to iob
    :param sequence: sequence to convert
    :return: converted sequence
    """

    # Fetching current labels
    source_labels = [item[-1] for item in sequence]
    target_labels = list()

    for i, label in enumerate(source_labels):

        # Fetching tag and label for current token
        if label.startswith("O"):
            current_tag = "O"
            current_cat = None
        else:
            if label.count('-') == 1:
                current_tag, current_cat = label.split("-")
            else:
                parts = label.split("-")
                current_tag = parts[0]
                current_cat = "-".join(parts[1:])

        next_tag, next_cat = None, None

        if i < len(source_labels) - 1:
            if source_labels[i+1].startswith("O"):
                next_tag = "O"
            else:
                if source_labels[i + 1].count('-') == 1:
                    next_tag, next_cat = source_labels[i+1].split("-")
                else:
                    parts = source_labels[i+1].split("-")
                    next_tag = parts[0]
                    next_cat = "-".join(parts[1:])

        previous_tag, previous_cat = None, None

        if i > 0:
            if source_labels[i-1].startswith("O"):
                previous_tag = "O"
            else:
                if source_labels[i - 1].count('-') == 1:
                    previous_tag, previous_cat = source_labels[i - 1].split("-")
                else:
                    parts = source_labels[i - 1].split("-")
                    previous_tag = parts[0]
                    previous_cat = "-".join(parts[1:])

        # Case where current tag is 'O', no change
        if current_tag == "O":
            target_labels.append(label)

        # Case where current tag is 'B'
        elif current_tag == "B":
            target_labels.append(label)

        # Case where current tag is 'I'
        elif current_tag == "I":
            target_labels.append(label)

        # Case where current tag is 'I'
        elif current_tag == "E":
            target_labels.append("{}-{}".format("I", current_cat))

        # Case where current tag is 'I'
        elif current_tag == "S":
            target_labels.append("{}-{}".format("B", current_cat))

    # Repacking sequence and returning
    final_bioes = list()

    for source_item, target_label in zip(sequence, target_labels):
        final_bioes.append(source_item[:-1] + [target_label])

    return final_bioes


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(title="Sub-commands", description="Valid sub-commands",
                                       help="Valid sub-commands", dest="subparser_name")

    parser_iob_to_iobes = subparsers.add_parser('IOB-TO-IOBES', help="Convert IOB to IOBES tagging scheme")

    parser_iob_to_iobes.add_argument("--input_file", help="input_file", dest="input_file", type=str, required=True)
    parser_iob_to_iobes.add_argument("--output_file", help="output_file", dest="output_file", type=str, required=True)

    parser_iobes_to_iob = subparsers.add_parser('IOBES-TO-IOB', help="Convert IOBES to IOB tagging scheme")

    parser_iobes_to_iob.add_argument("--input_file", help="input_file", dest="input_file", type=str, required=True)
    parser_iobes_to_iob.add_argument("--output_file", help="output_file", dest="output_file", type=str, required=True)

    args = parser.parse_args()

    if args.subparser_name == "IOB-TO-IOBES":

        # Logging to stdout
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

        # Checking if input file exists
        if not os.path.isfile(os.path.abspath(args.input_file)):
            raise FileNotFoundError("The input file already does not exist: {}".format(
                os.path.abspath(args.input_file)
            ))

        # Checking if output file exists
        if os.path.isfile(os.path.abspath(args.output_file)):
            raise FileExistsError("The output file you specified already exists {}".format(
                os.path.abspath(args.output_file)
            ))

        logging.info("Starting converting tagging scheme")
        logging.info("* source file: {}".format(os.path.abspath(args.input_file)))
        logging.info("* target file: {}".format(os.path.abspath(args.output_file)))

        start = time.time()

        iob_to_iobes(args.input_file, args.output_file)

        end = time.time()

        logging.info("Done ! (Time elapsed: {})".format(timedelta(seconds=round(end-start))))

    elif args.subparser_name == "IOBES-TO-IOB":

        # Logging to stdout
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

        # Checking if input file exists
        if not os.path.isfile(os.path.abspath(args.input_file)):
            raise FileNotFoundError("The input file already does not exist: {}".format(
                os.path.abspath(args.input_file)
            ))

        # Checking if output file exists
        if os.path.isfile(os.path.abspath(args.output_file)):
            raise FileExistsError("The output file you specified already exists {}".format(
                os.path.abspath(args.output_file)
            ))

        logging.info("Starting converting tagging scheme")
        logging.info("* source file: {}".format(os.path.abspath(args.input_file)))
        logging.info("* target file: {}".format(os.path.abspath(args.output_file)))

        start = time.time()

        iobes_to_iob(args.input_file, args.output_file)

        end = time.time()

        logging.info("Done ! (Time elapsed: {})".format(timedelta(seconds=round(end-start))))
