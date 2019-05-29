import re


def convert_spaces_to_tabulations(input_file: str = None,
                                  output_file: str = None) -> None:
    """
    Convert a CoNLL file with spaces as column separators into a CoNLL file with tabulations as column separators

    Args:
        input_file (str): input CoNLL filepath
        output_file (str): output CoNLL filepath

    Returns:
        None
    """

    with open(input_file, "r", encoding="UTF-8") as input_file:
        with open(output_file, "w", encoding="UTF-8") as output_file:
            for line in input_file:
                if re.match(r"^$", line):
                    output_file.write(line)
                    continue

                parts = line.rstrip("\n").split(" ")
                output_file.write("{}\n".format("\t".join(parts)))


def convert_labels(input_file: str = None,
                   output_file: str = None,
                   input_label_type: str = None,
                   output_label_type: str = None):
    """
    Convert NER tagging schemes

    Args:
        input_file (str): input CoNLL filepath
        output_file (str): output CoNLL filepath
        input_label_type (str): source NER tagging scheme
        output_label_type (str): target NER tagging scheme

    Returns:
        None
    """

    with open(input_file, "r", encoding="UTF-8") as input_file:
        with open(output_file, "w", encoding="UTF-8") as output_file:

            sequence_buffer = list()
            for line in input_file:
                if re.match(r"^$", line):
                    if len(sequence_buffer) > 0:
                        output_labels = convert_sequence(input_sequence=sequence_buffer,
                                                         input_label_type=input_label_type,
                                                         output_label_type=output_label_type)
                        for input_token, label in zip(sequence_buffer, output_labels):
                            final_parts = input_token[:-1] + [label]
                            output_file.write("{}\n".format("\t".join(final_parts)))
                            sequence_buffer = list()
                    output_file.write("\n")
                    continue

                sequence_buffer.append(line.rstrip("\n").split("\t"))

            if len(sequence_buffer) > 0:
                output_labels = convert_sequence(input_sequence=sequence_buffer,
                                                 input_label_type=input_label_type,
                                                 output_label_type=output_label_type)
                for input_token, label in zip(sequence_buffer, output_labels):
                    final_parts = input_token[:-1] + [label]
                    output_file.write("{}\n".format("\t".join(final_parts)))


def convert_sequence(input_sequence: list = None,
                     input_label_type: str = None,
                     output_label_type: str = None):

    input_labels = [token[-1] for token in input_sequence]

    if input_label_type == "CONLL2003":
        entities = extract_entities_conll2003(input_labels=input_labels)
    else:
        raise Exception

    output_labels = ["O"] * len(input_sequence)

    if output_label_type == "BIOUL":
        for entity in entities:
            if len(entity) == 1:
                output_labels[entity[0]] = "U-{}".format(input_labels[entity[0]].split("-")[1])
            else:
                for i, tok_idx in enumerate(entity):
                    if i == 0:
                        output_labels[tok_idx] = "B-{}".format(input_labels[tok_idx].split("-")[1])
                    elif i == len(entity) - 1:
                        output_labels[tok_idx] = "L-{}".format(input_labels[tok_idx].split("-")[1])
                    else:
                        output_labels[tok_idx] = "I-{}".format(input_labels[tok_idx].split("-")[1])

    else:
        raise Exception("Unrecognised output format")

    return output_labels


def extract_tag_cat(label):
    """
    Separate tag from category

    Args:
        label (str): NER label to split

    Returns:
        (str, str): tag, category
    """
    if label == "O":
        return "O", None
    else:
        return label.split("-")


def extract_entities_conll2003(input_labels: list = None):
    """
    Extract entity offsets for a CoNLL file encoded in conll 2003

    Args:
        input_labels (list): source labels

    Returns:
        list: entity offsets
    """

    sequence_entities = list()

    entity_buffer = list()
    for i, label in enumerate(input_labels):
        if i > 0:
            tag_before, cat_before = extract_tag_cat(label)
        else:
            tag_before, cat_before = None, None

        if label == "O":
            if len(entity_buffer) > 0:
                sequence_entities.append(entity_buffer)
                entity_buffer = list()
            continue

        tag, cat = label.split("-")

        if tag == "B":
            if len(entity_buffer) > 0:
                sequence_entities.append(entity_buffer)
                entity_buffer = list()

            entity_buffer.append(i)

        elif tag == "I":
            if tag_before == "O" or tag_before is None:
                entity_buffer.append(i)

            elif tag_before == "B":
                if cat_before == cat:
                    entity_buffer.append(i)
                else:
                    if len(entity_buffer) > 0:
                        sequence_entities.append(entity_buffer)
                        entity_buffer = list()
                    entity_buffer.append(i)

            elif tag_before == "I":
                if cat_before == cat:
                    entity_buffer.append(i)
                else:
                    if len(entity_buffer) > 0:
                        sequence_entities.append(entity_buffer)
                        entity_buffer = list()
                    entity_buffer.append(i)

    if len(entity_buffer) > 0:
        sequence_entities.append(entity_buffer)

    return sequence_entities
