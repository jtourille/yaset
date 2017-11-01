import argparse
import logging
import os
import re
import time
from datetime import timedelta


def check_tagging_scheme(input_file, scheme):
    """
    Check tagging scheme integrity in one CoNLL file.
    Accepted schemes: IOB, IOBE, IOBES.
    :param input_file: CoNLL file path
    :param scheme: tagging scheme applied in the input CoNLL file
    :return: True if verification is passed, throw Exceptions in other cases (boolean)
    """

    # Will contain all sequences from the input file
    all_sequences = list()

    # Sequence ID tracker
    seq_id = 1

    with open(os.path.abspath(input_file), "r", encoding="UTF-8") as input_file:
        current_sequence = list()

        # Gathering sequences
        for i, line in enumerate(input_file, start=1):
            if re.match("^$", line):
                if len(current_sequence) > 0:
                    all_sequences.append({
                        "seq_id": seq_id,
                        "seq_tokens": current_sequence.copy()
                    })

                    seq_id += 1
                    current_sequence.clear()

                continue

            # Line with tokens are split and stored
            parts = line.rstrip("\n").split("\t")
            current_sequence.append(parts)

        # End of file, adding information about the last sequence if necessary
        if len(current_sequence) > 0:
            all_sequences.append({
                "seq_id": seq_id,
                "seq_tokens": current_sequence.copy()
            })

    # Looping through sentences checking tagging scheme
    for sequence in all_sequences:
        _check_sequence(sequence["seq_id"], sequence["seq_tokens"], scheme)

    return True


def _seq_to_string(seq_tokens):
    """
    Return a string version of a sequence.
    :param seq_tokens: sequence to convert to string
    :return: string version of the sequence (string)
    """

    return " ".join(["{}/{}".format(parts[0], parts[-1]) for parts in seq_tokens])


def raise_tag_exception(seq_id, seq_tokens, cur=None, fol=None):
    """
    Helper to raise an Exception when two following tags are not authorized
    :param seq_id: sequence ID
    :param seq_tokens: sequence tokens
    :param cur: tag at position n
    :param fol: tag at position n+1
    :return: nothing
    """

    raise Exception("a tag {} follows a tag {} in seq #{}: {}".format(
        fol,
        cur,
        seq_id,
        _seq_to_string(seq_tokens)
    ))


def _check_sequence(seq_id, seq_tokens, scheme):
    """
    Check tagging scheme for one sequence
    :param seq_id: sequence ID used for Exception throwing (int)
    :param seq_tokens: sequence tokens (list of lists)
    :param scheme: sequence tagging scheme
    :return: nothing, will throw Exceptions if needed
    """

    # Fetching current labels
    source_labels = [item[-1] for item in seq_tokens]

    for i, label in enumerate(source_labels):

        # -----------------------------------------------------------
        # CURRENT

        current_tag, current_cat = None, None

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

        # -----------------------------------------------------------
        # NEXT

        next_tag, next_cat = None, None

        if i < len(source_labels) - 1:
            if source_labels[i + 1].startswith("O"):
                next_tag = "O"
                next_cat = None
            else:
                if source_labels[i + 1].count('-') == 1:
                    next_tag, next_cat = source_labels[i + 1].split("-")
                else:
                    parts = source_labels[i + 1].split("-")
                    next_tag = parts[0]
                    next_cat = "-".join(parts[1:])

        # -----------------------------------------------------------
        # PREVIOUS

        previous_tag, previous_cat = None, None

        if i > 0:
            if source_labels[i - 1].startswith("O"):
                previous_tag = "O"
                previous_cat = None
            else:
                if source_labels[i - 1].count('-') == 1:
                    previous_tag, previous_cat = source_labels[i - 1].split("-")
                else:
                    parts = source_labels[i - 1].split("-")
                    previous_tag = parts[0]
                    previous_cat = "-".join(parts[1:])

        # -----------------------------------------------------------
        # TAG CHECKING

        if not _check_tag(current_tag, scheme):
            raise Exception("The tag for token #{} in seq #{} is not a valid tag: {}".format(
                i + 1,
                seq_id,
                "{}-{}".format(current_tag, current_cat)
            ))

        # -----------------------------------------------------------
        # LABEL 'O'

        if current_tag == "O":

            if scheme == "IOB":

                if next_tag is not None and next_tag in ["I"]:
                    raise_tag_exception(seq_id, seq_tokens, cur=current_tag, fol=next_tag)

            elif scheme == "IOBE":

                if previous_tag is not None and previous_tag in ["I"]:
                    raise_tag_exception(seq_id, seq_tokens, cur=previous_tag, fol=current_tag)

                if next_tag is not None and next_tag in ["I", "E"]:
                    raise_tag_exception(seq_id, seq_tokens, cur=current_tag, fol=next_tag)

            elif scheme == "IOBES":

                if previous_tag is not None and previous_tag in ["I", "B"]:
                    raise_tag_exception(seq_id, seq_tokens, cur=previous_tag, fol=current_tag)

                if next_tag is not None and next_tag in ["I", "E"]:
                    raise_tag_exception(seq_id, seq_tokens, cur=current_tag, fol=next_tag)

        # -----------------------------------------------------------
        # LABEL 'B'

        elif current_tag == "B":

            if scheme == "IOB":

                pass

            elif scheme == "IOBE":

                if previous_tag is not None and previous_tag in ["I"]:
                    raise_tag_exception(seq_id, seq_tokens, cur=previous_tag, fol=current_tag)

                if not next_tag:
                    raise Exception("last token has a B tag in seq #{}: {}".format(
                        seq_id,
                        _seq_to_string(seq_tokens)
                    ))

            elif scheme == "IOBES":

                if previous_tag is not None and previous_tag in ["I", "B"]:
                    raise_tag_exception(seq_id, seq_tokens, cur=previous_tag, fol=current_tag)

                if next_tag is not None and next_tag in ["O", "B", "S"]:
                    raise_tag_exception(seq_id, seq_tokens, cur=current_tag, fol=next_tag)

                if not next_tag:
                    raise Exception("last token has a B tag in seq #{}: {}".format(
                        seq_id,
                        _seq_to_string(seq_tokens)
                    ))

        # -----------------------------------------------------------
        # LABEL 'I'

        elif current_tag == "I":

            if not previous_tag:
                raise Exception("first token has a I tag in seq #{}: {}".format(
                    seq_id,
                    _seq_to_string(seq_tokens)
                ))

            if scheme == "IOB":

                if previous_tag is not None and previous_tag in ["O"]:
                    raise_tag_exception(seq_id, seq_tokens, cur=previous_tag, fol=current_tag)

                if previous_tag is not None and previous_tag in ["B", "I"]:
                    if previous_cat != current_cat:
                        raise Exception("A tag I follows a tag {} with incompatible category in seq #{}: {}".format(
                            previous_tag,
                            seq_id,
                            _seq_to_string(seq_tokens)
                        ))

            elif scheme == "IOBE":

                if previous_tag is not None and previous_tag in ["O", "E"]:
                    raise_tag_exception(seq_id, seq_tokens, cur=previous_tag, fol=current_tag)

                if next_tag is not None and next_tag in ["O", "B"]:
                    raise_tag_exception(seq_id, seq_tokens, cur=current_tag, fol=next_tag)

                if previous_tag is not None and previous_tag in ["B", "I"]:
                    if previous_cat != current_cat:
                        raise Exception("A tag I follows a tag {} with incompatible category in seq #{}: {}".format(
                            previous_tag,
                            seq_id,
                            _seq_to_string(seq_tokens)
                        ))

                if not next_tag:
                    raise Exception("last token has a I tag in seq #{}: {}".format(
                        seq_id,
                        _seq_to_string(seq_tokens)
                    ))

            elif scheme == "IOBES":

                if previous_tag is not None and previous_tag in ["O", "E", "S"]:
                    raise_tag_exception(seq_id, seq_tokens, cur=previous_tag, fol=current_tag)

                if next_tag is not None and next_tag in ["O", "B", "S"]:
                    raise_tag_exception(seq_id, seq_tokens, cur=current_tag, fol=next_tag)

                if previous_tag is not None and previous_tag in ["B", "I"]:
                    if previous_cat != current_cat:
                        raise Exception("A tag I follows a tag {} with incompatible category in seq #{}: {}".format(
                            previous_tag,
                            seq_id,
                            _seq_to_string(seq_tokens)
                        ))

                if not next_tag:
                    raise Exception("last token has a I tag in seq #{}: {}".format(
                        seq_id,
                        _seq_to_string(seq_tokens)
                    ))

        # -----------------------------------------------------------
        # LABEL 'E'

        elif current_tag == "E":

            if not previous_tag:
                raise Exception("first token has a E tag in seq #{}: {}".format(
                    seq_id,
                    _seq_to_string(seq_tokens)
                ))

            if scheme == "IOBE":

                if previous_tag is not None and previous_tag in ["O", "E"]:
                    raise_tag_exception(seq_id, seq_tokens, cur=previous_tag, fol=current_tag)

                if next_tag is not None and next_tag in ["I", "E"]:
                    raise_tag_exception(seq_id, seq_tokens, cur=current_tag, fol=next_tag)

                if previous_tag is not None and previous_tag in ["B", "I"]:
                    if previous_cat != current_cat:
                        raise Exception("A tag E follows a tag {} with incompatible category in seq #{}: {}".format(
                            previous_tag,
                            seq_id,
                            _seq_to_string(seq_tokens)
                        ))

            if scheme == "IOBES":

                if previous_tag is not None and previous_tag in ["O", "E", "S"]:
                    raise_tag_exception(seq_id, seq_tokens, cur=previous_tag, fol=current_tag)

                if next_tag is not None and next_tag in ["I", "E"]:
                    raise_tag_exception(seq_id, seq_tokens, cur=current_tag, fol=next_tag)

                if previous_tag is not None and previous_tag in ["B", "I"]:
                    if previous_cat != current_cat:
                        raise Exception("A tag E follows a tag {} with incompatible category in seq #{}: {}".format(
                            previous_tag,
                            seq_id,
                            _seq_to_string(seq_tokens)
                        ))

        # -----------------------------------------------------------
        # LABEL 'S'

        elif current_tag == "S":

            if scheme == "IOBES":

                if previous_tag is not None and previous_tag in ["I", "B"]:
                    raise_tag_exception(seq_id, seq_tokens, cur=previous_tag, fol=current_tag)

                if next_tag is not None and next_tag in ["I", "E"]:
                    raise_tag_exception(seq_id, seq_tokens, cur=current_tag, fol=next_tag)


def _check_tag(tag, scheme):
    """
    Check a given token tag against the list of authorized tags for the tagging scheme
    :param tag: tag to check
    :param scheme:
    :return:
    """

    if scheme == "IOB":
        if tag not in ["I", "O", "B"]:
            return False
        else:
            return True

    elif scheme == "IOBE":
        if tag not in ["I", "O", "B", "E"]:
            return False
        else:
            return True

    elif scheme == "IOBES":
        if tag not in ["I", "O", "B", "E", "S"]:
            return False
        else:
            return True

    else:
        raise Exception("The tagging scheme you specified does not exists or is not supported: {}".format(
            scheme
        ))


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
    entities = list()

    current_entity_cat = None
    current_entity_tokens = list()

    for i, label in enumerate(source_labels):

        # -----------------------------------------------------------
        # CURRENT

        current_tag, current_cat = None, None

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

        # -----------------------------------------------------------
        # NEXT

        next_tag, next_cat = None, None

        if i < len(source_labels) - 1:
            if source_labels[i + 1].startswith("O"):
                next_tag = "O"
                next_cat = None
            else:
                if source_labels[i + 1].count('-') == 1:
                    next_tag, next_cat = source_labels[i + 1].split("-")
                else:
                    parts = source_labels[i + 1].split("-")
                    next_tag = parts[0]
                    next_cat = "-".join(parts[1:])

        # -----------------------------------------------------------
        # PREVIOUS

        previous_tag, previous_cat = None, None

        if i > 0:
            if source_labels[i - 1].startswith("O"):
                previous_tag = "O"
                previous_cat = None
            else:
                if source_labels[i - 1].count('-') == 1:
                    previous_tag, previous_cat = source_labels[i - 1].split("-")
                else:
                    parts = source_labels[i - 1].split("-")
                    previous_tag = parts[0]
                    previous_cat = "-".join(parts[1:])

        if current_tag in ["B"]:
            if previous_tag in ["I"]:

                # Clearing entity
                new_entity = "{}##{}".format(current_entity_cat,
                                             "-".join([str(item) for item in sorted(current_entity_tokens)]))
                entities.append(new_entity)

                current_entity_cat = None
                current_entity_tokens.clear()

                # Starting new entity
                current_entity_cat = current_cat
                current_entity_tokens.append(i)

            elif previous_tag in ["B"]:

                # Clearing entity
                new_entity = "{}##{}".format(current_entity_cat,
                                             "-".join([str(item) for item in sorted(current_entity_tokens)]))
                entities.append(new_entity)

                current_entity_cat = None
                current_entity_tokens.clear()

                # Starting new entity
                current_entity_cat = current_cat
                current_entity_tokens.append(i)

            elif previous_tag in ["O", None]:

                # Starting new entity
                current_entity_cat = current_cat
                current_entity_tokens.append(i)

        if current_tag in ["O"]:
            if previous_tag in ["B", "I"]:

                # Clearing entity
                new_entity = "{}##{}".format(current_entity_cat,
                                             "-".join([str(item) for item in sorted(current_entity_tokens)]))
                entities.append(new_entity)

                current_entity_cat = None
                current_entity_tokens.clear()

        if current_tag in ["I"]:
            if previous_tag in ["O"]:
                # Starting new entity
                current_entity_cat = current_cat
                current_entity_tokens.append(i)

            elif previous_tag in ["B"]:

                if current_cat != current_entity_cat:

                    # Clearing entity
                    new_entity = "{}##{}".format(current_entity_cat,
                                                 "-".join([str(item) for item in sorted(current_entity_tokens)]))
                    entities.append(new_entity)

                    current_entity_cat = None
                    current_entity_tokens.clear()

                    # Starting new entity
                    current_entity_cat = current_cat
                    current_entity_tokens.append(i)

                else:
                    current_entity_tokens.append(i)

            elif previous_tag in ["I"]:

                if current_cat != current_entity_cat:

                    # Clearing entity
                    new_entity = "{}##{}".format(current_entity_cat,
                                                 "-".join([str(item) for item in sorted(current_entity_tokens)]))
                    entities.append(new_entity)

                    current_entity_cat = None
                    current_entity_tokens.clear()

                    # Starting new entity
                    current_entity_cat = current_cat
                    current_entity_tokens.append(i)
                else:
                    current_entity_tokens.append(i)

    if len(current_entity_tokens) > 0:

        # Clearing entity
        new_entity = "{}##{}".format(current_entity_cat,
                                     "-".join([str(item) for item in sorted(current_entity_tokens)]))
        entities.append(new_entity)

    done = set()
    target_labels = ["O" for _ in sequence]

    for entity in entities:

        entity_cat, entity_tokens = entity.split("##")
        token_id = [int(i) for i in entity_tokens.split("-")]

        for item in token_id:
            if item in done:
                raise Exception("Overlapping entities")
            done.add(item)

        if len(token_id) == 1:
            target_labels[token_id[0]] = "S-{}".format(entity_cat)
        else:
            target_labels[token_id[0]] = "B-{}".format(entity_cat)
            target_labels[token_id[-1]] = "E-{}".format(entity_cat)
            for index in token_id[1:-1]:
                target_labels[index] = "I-{}".format(entity_cat)

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
    entities = list()

    current_entity_cat = None
    current_entity_tokens = list()

    previous_tag = None

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

        if current_tag in ["B", "S"]:

            if previous_tag in ["I", "B", "E", "S"]:
                # Clearing entity
                new_entity = "{}##{}".format(current_entity_cat,
                                             "-".join([str(item) for item in sorted(current_entity_tokens)]))
                entities.append(new_entity)

                current_entity_tokens.clear()

                # Starting new entity
                current_entity_cat = current_cat
                current_entity_tokens.append(i)

                previous_tag = "B"

            elif previous_tag in ["O", None]:
                # Starting new entity
                current_entity_cat = current_cat
                current_entity_tokens.append(i)

                previous_tag = "B"

            else:
                raise Exception("Something wrong happened")

        elif current_tag in ["O"]:

            if previous_tag in ["I", "B", "E", "S"]:

                # Clearing entity
                new_entity = "{}##{}".format(current_entity_cat,
                                             "-".join([str(item) for item in sorted(current_entity_tokens)]))
                entities.append(new_entity)

                current_entity_cat = None
                current_entity_tokens.clear()

                previous_tag, previous_cat = "O", None

        elif current_tag in ["I", "E"]:

            if previous_tag in ["E", "S"]:

                # Clearing entity
                new_entity = "{}##{}".format(current_entity_cat,
                                             "-".join([str(item) for item in sorted(current_entity_tokens)]))
                entities.append(new_entity)

                current_entity_tokens.clear()

                # Starting new entity
                current_entity_cat = current_cat
                current_entity_tokens.append(i)

                previous_tag = "B"

            elif previous_tag in ["O", None]:
                # Starting new entity
                current_entity_cat = current_cat
                current_entity_tokens.append(i)

                previous_tag = "B"

            elif previous_tag in ["I", "B"]:

                if current_cat != current_entity_cat:

                    # Clearing entity
                    new_entity = "{}##{}".format(current_entity_cat,
                                                 "-".join([str(item) for item in sorted(current_entity_tokens)]))
                    entities.append(new_entity)

                    current_entity_tokens.clear()

                    # Starting new entity
                    current_entity_cat = current_cat
                    current_entity_tokens.append(i)
                    
                    previous_tag = "B"

                else:

                    previous_tag = "I"
                    current_entity_tokens.append(i)

    if len(current_entity_tokens) > 0:

        # Clearing entity
        new_entity = "{}##{}".format(current_entity_cat,
                                     "-".join([str(item) for item in sorted(current_entity_tokens)]))
        entities.append(new_entity)

    done = set()
    target_labels = ["O" for _ in sequence]

    for entity in entities:

        entity_cat, entity_tokens = entity.split("##")
        token_id = [int(i) for i in entity_tokens.split("-")]

        for item in token_id:
            if item in done:
                raise Exception("Overlapping entities")
            done.add(item)

        if len(token_id) == 1:
            target_labels[token_id[0]] = "B-{}".format(entity_cat)
        else:
            target_labels[token_id[0]] = "B-{}".format(entity_cat)
            for index in token_id[1:]:
                target_labels[index] = "I-{}".format(entity_cat)

    final_bioes = list()

    for source_item, target_label in zip(sequence, target_labels):
        final_bioes.append(source_item[:-1] + [target_label])

    return final_bioes


def convert_to_one_class(source_file, target_file):
    """
    Convert multi-class IOBES scheme into one-class IOBES scheme
    :param source_file: input file path
    :param target_file: target file path
    :return:
    """

    with open(os.path.abspath(source_file), "r", encoding="UTF-8") as input_file:
        with open(os.path.abspath(target_file), "w", encoding="UTF-8") as output_file:
            for i, line in enumerate(input_file, start=1):
                if re.match("^$", line):
                    output_file.write("\n")
                    continue

                parts = line.rstrip("\n").split('\t')
                if parts[-1].startswith("I"):
                    parts[-1] = "I-UNK"

                elif parts[-1].startswith("B"):
                    parts[-1] = "B-UNK"

                elif parts[-1].startswith("E"):
                    parts[-1] = "E-UNK"

                elif parts[-1].startswith("S"):
                    parts[-1] = "S-UNK"

                elif parts[-1].startswith("O"):
                    pass

                else:
                    raise Exception("Label should start with I, O, B, E or S, got {} at line {}: {}".format(
                        parts[-1][0], i, line.rstrip("\n")
                    ))

                output_file.write("{}\n".format("\t".join(parts)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(title="Sub-commands", description="Valid sub-commands",
                                       help="Valid sub-commands", dest="subparser_name")

    # Convert IOB tagging scheme to IOBES
    parser_iob_to_iobes = subparsers.add_parser('IOB-TO-IOBES', help="Convert IOB to IOBES tagging scheme")
    parser_iob_to_iobes.add_argument("--input-file", help="Input CoNLL file to convert", dest="input_file",
                                     type=str, required=True)
    parser_iob_to_iobes.add_argument("--output-file", help="Output CoNLL file", dest="output_file", type=str,
                                     required=True)
    parser_iob_to_iobes.add_argument('--check-integrity', help="Check tagging scheme integrity", dest="check",
                                     action='store_true')

    # Convert IOBES tagging scheme to IOB
    parser_iobes_to_iob = subparsers.add_parser('IOBES-TO-IOB', help="Convert IOBES to IOB tagging scheme")
    parser_iobes_to_iob.add_argument("--input-file", help="Input CoNLL file to convert", dest="input_file", type=str,
                                     required=True)
    parser_iobes_to_iob.add_argument("--output-file", help="Output CoNLL file", dest="output_file", type=str,
                                     required=True)
    parser_iobes_to_iob.add_argument('--check-integrity', help="Check tagging scheme integrity", dest="check",
                                     action='store_true')

    parser_check_format = subparsers.add_parser('CHECK', help="Check tagging scheme coherence")
    parser_check_format.add_argument("--input-file", help="Input CoNLL file to check", dest="input_file",
                                     type=str, required=True)

    group_scheme = parser_check_format.add_mutually_exclusive_group(required=True)
    group_scheme.add_argument('--iob', action='store_true')
    group_scheme.add_argument('--iobe', action='store_true')
    group_scheme.add_argument('--iobes', action='store_true')

    # Multiclass to oneclass
    parser_multi_to_one = subparsers.add_parser('MULTI-TO-ONE', help="Convert IOBES to IOB tagging scheme")
    parser_multi_to_one.add_argument("--input-file", help="Input CoNLL file to convert", dest="input_file", type=str,
                                     required=True)
    parser_multi_to_one.add_argument("--output-file", help="Output CoNLL file", dest="output_file", type=str,
                                     required=True)

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
        logging.info("* check tagging scheme integrity: {}".format(args.check))

        start = time.time()

        if args.check:
            logging.info("Checking tagging scheme integrity")
            check_tagging_scheme(args.input_file, "IOB")

        logging.info("Converting file")
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
        logging.info("* check tagging scheme integrity: {}".format(args.check))

        start = time.time()

        if args.check:
            logging.info("Checking tagging scheme integrity")
            check_tagging_scheme(args.input_file, "IOBES")

        logging.info("Converting file")
        iobes_to_iob(args.input_file, args.output_file)

        end = time.time()

        logging.info("Done ! (Time elapsed: {})".format(timedelta(seconds=round(end-start))))

    elif args.subparser_name == "MULTI-TO-ONE":

        # Logging to stdout
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

        # Checking if input file exists
        if not os.path.isfile(os.path.abspath(args.input_file)):
            raise FileNotFoundError("The input file does not exist: {}".format(
                os.path.abspath(args.input_file)
            ))

        # Checking if output file exists
        if os.path.isfile(os.path.abspath(args.output_file)):
            raise FileExistsError("The output file you specified already exists {}".format(
                os.path.abspath(args.output_file)
            ))

        logging.info("Starting conversion")
        logging.info("* source file: {}".format(os.path.abspath(args.input_file)))
        logging.info("* target file: {}".format(os.path.abspath(args.output_file)))

        start = time.time()

        logging.info("Converting file")
        convert_to_one_class(args.input_file, args.output_file)

        end = time.time()

        logging.info("Done ! (Time elapsed: {})".format(timedelta(seconds=round(end - start))))

    elif args.subparser_name == "CHECK":

        # Logging to stdout
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

        # Checking if input file exists
        if not os.path.isfile(os.path.abspath(args.input_file)):
            raise FileNotFoundError("The input file already does not exist: {}".format(
                os.path.abspath(args.input_file)
            ))

        tag_scheme = None

        if args.iob:
            tag_scheme = "IOB"
        elif args.iobe:
            tag_scheme = "IOBE"
        elif args.iobes:
            tag_scheme = "IOBES"

        logging.info("Starting checking tagging scheme")
        logging.info("* source file: {}".format(os.path.abspath(args.input_file)))
        logging.info("* tagging scheme: {}".format(tag_scheme))

        start = time.time()

        check_tagging_scheme(args.input_file, tag_scheme)

        end = time.time()

        logging.info("Done ! (Time elapsed: {})".format(timedelta(seconds=round(end - start))))
