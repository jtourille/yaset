import sys
from collections import defaultdict, namedtuple

from prettytable import PrettyTable

Metrics = namedtuple('Metrics', 'tp fp fn prec rec fscore')


class EvalCounts(object):

    def __init__(self):

        self.correct_chunk = 0    # number of correctly identified chunks
        self.correct_tags = 0     # number of correct chunk tags
        self.found_correct = 0    # number of chunks in corpus
        self.found_guessed = 0    # number of identified chunks
        self.token_counter = 0    # token counter (ignores sentence breaks)

        # counts by type
        self.t_correct_chunk = defaultdict(int)
        self.t_found_correct = defaultdict(int)
        self.t_found_guessed = defaultdict(int)


def parse_label(tag):
    """
    Parse a IOBES tag and return the tag label and category
    :param tag: tag to parse
    :return:
    """

    if tag.startswith("O"):
        current_tag = "O"
        current_cat = None
    else:
        if tag.count('-') == 1:
            current_tag, current_cat = tag.split("-")
        else:
            parts = tag.split("-")
            current_tag = parts[0]
            current_cat = "-".join(parts[1:])

    return current_tag, current_cat


def evaluate(sequences):
    """
    Compute conll metrics over a set of sequences
    :param sequences: sequences
    :return: EvalCounts object
    """

    counts = EvalCounts()

    for sequence in sequences:

        in_correct = False  # currently processed chunks is correct until now

        last_correct = 'O'  # previous chunk tag in corpus
        last_correct_type = ''  # type of previously identified chunk tag

        last_guessed = 'O'  # previously identified chunk tag
        last_guessed_type = ''  # type of previous chunk tag in corpus

        pred_labels = [item["pred"] for item in sequence]
        gs_labels = [item["gs"] for item in sequence]

        pred_labels_corrected = _iobes_to_iob_seq(pred_labels)
        gs_labels_corrected = _iobes_to_iob_seq(gs_labels)

        for pred, gs in zip(pred_labels_corrected, gs_labels_corrected):

            guessed, guessed_type = parse_label(pred)
            correct, correct_type = parse_label(gs)

            end_correct = end_of_chunk(last_correct, correct, last_correct_type, correct_type)
            end_guessed = end_of_chunk(last_guessed, guessed, last_guessed_type, guessed_type)

            start_correct = start_of_chunk(last_correct, correct, last_correct_type, correct_type)
            start_guessed = start_of_chunk(last_guessed, guessed, last_guessed_type, guessed_type)

            if in_correct:
                if end_correct and end_guessed and last_guessed_type == last_correct_type:
                    in_correct = False
                    counts.correct_chunk += 1
                    counts.t_correct_chunk[last_correct_type] += 1

                elif end_correct != end_guessed or guessed_type != correct_type:
                    in_correct = False

            if start_correct and start_guessed and guessed_type == correct_type:
                in_correct = True

            if start_correct:
                counts.found_correct += 1
                counts.t_found_correct[correct_type] += 1

            if start_guessed:
                counts.found_guessed += 1
                counts.t_found_guessed[guessed_type] += 1

            if correct == guessed and guessed_type == correct_type:
                counts.correct_tags += 1

            counts.token_counter += 1

            last_guessed = guessed
            last_correct = correct
            last_guessed_type = guessed_type
            last_correct_type = correct_type

        if in_correct:
            counts.correct_chunk += 1
            counts.t_correct_chunk[last_correct_type] += 1

    return counts


def calculate_metrics(correct, guessed, total):
    """
    Compute precision, recall and f1-measure
    :param correct: number of correctly predicted entities
    :param guessed: number of predicted entities
    :param total: number of gold-standard entities
    :return: namedtuple
    """

    tp, fp, fn = correct, guessed - correct, total - correct

    p = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f = 0 if p + r == 0 else 2 * p * r / (p + r)

    return Metrics(tp, fp, fn, p, r, f)


def metrics(counts):
    """
    Compute metrics for each category and overall
    :param counts:
    :return:
    """

    c = counts
    overall = calculate_metrics(c.correct_chunk, c.found_guessed, c.found_correct)
    by_type = dict()

    k_list = set()
    for item in c.t_found_correct.keys():
        k_list.add(item)

    for item in c.t_found_guessed.keys():
        k_list.add(item)

    for t in list(k_list):
        by_type[t] = calculate_metrics(
            c.t_correct_chunk[t], c.t_found_guessed[t], c.t_found_correct[t]
        )

    return overall, by_type


def build_report(counts):

    overall, by_type = metrics(counts)

    x = PrettyTable()
    x.field_names = ["Category", "P", "R", "F1"]

    x.add_row(["OVERALL", "{:02.2f}".format((100. * overall.prec)), "{:02.2f}".format((100. * overall.rec)),
               "{:02.2f}".format((100. * overall.fscore))])

    for i, m in sorted(by_type.items()):
        x.add_row([i, "{:02.2f}".format((100. * m.prec)), "{:02.2f}".format((100. * m.rec)),
                   "{:02.2f}".format((100. * m.fscore))])

    x.align["Category"] = "r"

    return x


def report_old(counts, out=None):

    if out is None:
        out = sys.stdout

    overall, by_type = metrics(counts)

    c = counts
    out.write('processed %d tokens with %d phrases; ' %
              (c.token_counter, c.found_correct))
    out.write('found: %d phrases; correct: %d.\n' %
              (c.found_guessed, c.correct_chunk))

    if c.token_counter > 0:
        out.write('accuracy: %6.2f%%; ' %
                  (100.*c.correct_tags/c.token_counter))
        out.write('precision: %6.2f%%; ' % (100.*overall.prec))
        out.write('recall: %6.2f%%; ' % (100.*overall.rec))
        out.write('FB1: %6.2f\n' % (100.*overall.fscore))

    for i, m in sorted(by_type.items()):
        out.write('%50s: ' % i)
        out.write('precision: %6.2f%%; ' % (100.*m.prec))
        out.write('recall: %6.2f%%; ' % (100.*m.rec))
        out.write('FB1: %6.2f  %d\n' % (100.*m.fscore, c.t_found_guessed[i]))


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """
    Check if a chunk ended between the previous and current token.
    Capable of handling IOBES tagging scheme.
    :param prev_tag: previous tag
    :param tag: current tag
    :param prev_type: previous token category
    :param type_: current token category
    :return: True if chunk has ended, False otherwise
    """

    chunk_end = False

    if prev_tag == 'E':
        chunk_end = True

    if prev_tag == 'S':
        chunk_end = True

    if prev_tag == 'B' and tag == 'B':
        chunk_end = True

    if prev_tag == 'B' and tag == 'S':
        chunk_end = True

    if prev_tag == 'B' and tag == 'O':
        chunk_end = True

    if prev_tag == 'I' and tag == 'B':
        chunk_end = True

    if prev_tag == 'I' and tag == 'S':
        chunk_end = True

    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """
    Check if a chunk started between the previous and current word.
    Capable of handling IOBES tagging scheme.
    :param prev_tag: previous token tag
    :param tag: current token tag
    :param prev_type: previous token category
    :param type_: current token category
    :return: True if chunk has started, False otherwise
    """

    chunk_start = False

    if tag == 'B':
        chunk_start = True

    if tag == 'S':
        chunk_start = True

    if prev_tag == 'E' and tag == 'E':
        chunk_start = True

    if prev_tag == 'E' and tag == 'I':
        chunk_start = True

    if prev_tag == 'S' and tag == 'E':
        chunk_start = True

    if prev_tag == 'S' and tag == 'I':
        chunk_start = True

    if prev_tag == 'O' and tag == 'E':
        chunk_start = True

    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and prev_type != type_:
        chunk_start = True

    return chunk_start


def _iobes_to_iob_seq(source_labels):
    """
    Convert a sequence of IOB tags to IOBES
    :param source_labels: sequence of labels to convert
    :return: converted sequence
    """

    # Fetching current labels
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
    target_labels = ["O" for _ in source_labels]

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

    return target_labels
