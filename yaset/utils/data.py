import copy
import os
import random
import re
from collections import defaultdict

import torch
from allennlp.modules.elmo import batch_to_ids
from torch.utils.data import Dataset


class NERDatasetEnsemble(Dataset):

    def __init__(self,
                 model_mappings: dict = None,
                 instance_file: str = None,
                 testing: bool = None,
                 sentence_size_mapping: dict = None):

        self.mappings = model_mappings
        self.instance_file = instance_file
        self.testing = testing
        self.sentence_size_mapping = sentence_size_mapping

        self.instances = dict()

        self.load_instances()

    def __len__(self):

        if self.testing:
            return len(self.instances) // 10
        else:
            return len(self.instances)

    def __getitem__(self, idx):

        return self.instances[idx]

    def create_instance(self,
                        sequence_buffer: list = None):

        new_instance = defaultdict(dict)

        for model_id, model_mapping in self.mappings.items():
            model_instance = {
                "tok": list(),
                "str": list(),
                "chr_cnn": list(),
                "chr_lstm": list(),

                "lbl": list()
            }

            for item in sequence_buffer:
                token_form = item[0]
                token_label = item[-1]

                token_lower = token_form.lower()
                token_chr_cnn = [model_mapping["characters"].get("<bow>")] + \
                                [model_mapping["characters"].get(char) for char in token_form
                                 if model_mapping["characters"].get(char)] + \
                                [model_mapping["characters"].get("<eow>")]
                token_chr_lstm = [model_mapping["characters"].get(char) for char in token_form
                                  if model_mapping["characters"].get(char)]

                model_instance["tok"].append(
                    model_mapping["tokens"].get(token_lower, model_mapping["tokens"].get("<unk>"))
                )
                model_instance["str"].append(token_form)
                model_instance["chr_cnn"].append(token_chr_cnn)
                model_instance["chr_lstm"].append(token_chr_lstm)
                model_instance["lbl"].append(model_mapping["ner_labels"].get(token_label))

            for (lower, upper), sent_id in self.sentence_size_mapping.items():
                if lower <= len(model_instance["str"]) <= upper:
                    model_instance["sent_size"] = sent_id
                    break

            else:
                raise Exception("Problem")

            new_instance[model_id] = model_instance

        return new_instance

    def load_instances(self):

        instance_idx = 0

        with open(os.path.abspath(self.instance_file), "r", encoding="UTF-8") as input_file:
            sequence_buffer = list()
            for line in input_file:
                if re.match("^$", line):
                    if len(sequence_buffer) > 0:
                        instance = self.create_instance(sequence_buffer=sequence_buffer)
                        self.instances[instance_idx] = instance
                        sequence_buffer = list()
                        instance_idx += 1

                    continue

                sequence_buffer.append(line.rstrip("\n").split("\t"))

            if len(sequence_buffer) > 0:
                instance = self.create_instance(sequence_buffer=sequence_buffer)
                self.instances[instance_idx] = instance
                instance_idx += 1


def collate_ner_ensemble(batch,
                         model_mappings: dict = None,
                         model_options: dict = None,
                         reference_id: str = None):

    len_chr_lstm = list()
    len_char_cnn = list()
    len_tok = list()

    for instance in batch:
        for model_id, model_instance in instance.items():
            for item in model_instance["chr_lstm"]:
                len_chr_lstm.append(len(item))

            for item in model_instance["chr_cnn"]:
                len_char_cnn.append(len(item))

            len_tok.append(len(model_instance["chr_lstm"]))

    max_len_chr_lstm = max(len_chr_lstm)
    max_len_chr_cnn = max(len_char_cnn)
    max_len_tok = max(len_tok)

    final_batch = defaultdict(lambda: defaultdict(list))

    for instance in batch:
        for model_id, model_instance in instance.items():

            tok_pad_id = model_mappings[model_id].get("tokens").get("<pad>")
            chr_pad_id = model_mappings[model_id].get("characters").get("<pad>")

            # CHARS
            # =====

            chr_list_cnn = list()
            chr_list_lstm = list()
            chr_list_len = list()
            mask = list()

            for token_chars in model_instance["chr_lstm"]:
                cur_chars = copy.deepcopy(token_chars)
                chr_list_len.append(len(cur_chars))

                while len(cur_chars) < max_len_chr_lstm:
                    cur_chars.append(chr_pad_id)

                chr_list_lstm.append(cur_chars)

                mask.append(1)

            for token_chars in model_instance["chr_cnn"]:
                cur_chars = copy.deepcopy(token_chars)

                while len(cur_chars) < max_len_chr_cnn:
                    cur_chars.append(chr_pad_id)

                chr_list_cnn.append(cur_chars)

            while len(chr_list_cnn) < max_len_tok:
                cur_chars = [chr_pad_id for _ in range(max_len_chr_cnn)]
                chr_list_cnn.append(cur_chars)

                cur_chars = [chr_pad_id for _ in range(max_len_chr_lstm)]
                chr_list_lstm.append(cur_chars)

                chr_list_len.append(0)

                mask.append(0)

            final_batch[model_id]["chr_cnn"].append(chr_list_cnn)
            final_batch[model_id]["chr_lstm"].append(chr_list_lstm)
            final_batch[model_id]["chr_len"].append(chr_list_len)

            # STRINGS
            # =======

            final_batch[model_id]["str"].append(model_instance["str"])

            # TOKENS
            # ======

            final_batch[model_id]["tok_len"].append(len(model_instance["chr_lstm"]))

            if model_options.get(model_id).get("embeddings").get("pretrained").get("use"):
                cur_tokens = copy.deepcopy(model_instance["tok"])

                while len(cur_tokens) < max_len_tok:
                    cur_tokens.append(tok_pad_id)

                final_batch[model_id]["tok"].append(cur_tokens)

            # LABELS
            # ======

            cur_labels = copy.deepcopy(model_instance["lbl"])

            while len(cur_labels) < max_len_tok:
                cur_labels.append(0)

            final_batch[model_id]["labels"].append(cur_labels)
            final_batch[model_id]["mask"].append(mask)

            # SENTENCE SIZE
            # =============

            final_batch[model_id]["sent_size"].append(model_instance["sent_size"])

    final_batch_tensorised = dict()
    for model_id, model_batch in final_batch.items():
        final_batch_tensorised[model_id] = dict()

        final_batch_tensorised[model_id]["str"] = model_batch["str"]
        final_batch_tensorised[model_id]["elmo"] = batch_to_ids(model_batch["str"])

        final_batch_tensorised[model_id]["chr_lstm"] = torch.LongTensor(model_batch["chr_lstm"])
        final_batch_tensorised[model_id]["chr_cnn"] = torch.LongTensor(model_batch["chr_cnn"])
        final_batch_tensorised[model_id]["chr_len"] = torch.LongTensor(model_batch["chr_len"])

        final_batch_tensorised[model_id]["tok"] = torch.LongTensor(model_batch["tok"])
        final_batch_tensorised[model_id]["tok_len"] = torch.LongTensor(model_batch["tok_len"])

        final_batch_tensorised[model_id]["labels"] = torch.LongTensor(model_batch["labels"])
        final_batch_tensorised[model_id]["mask"] = torch.LongTensor(model_batch["mask"])

        final_batch_tensorised[model_id]["size"] = final_batch_tensorised[model_id]["chr_lstm"].size(0)

    final_batch_tensorised["labels"] = torch.LongTensor(final_batch[reference_id]["labels"])
    final_batch_tensorised["sent_size"] = torch.LongTensor(final_batch[reference_id]["sent_size"])
    final_batch_tensorised["mask"] = torch.LongTensor(final_batch[reference_id]["mask"])
    final_batch_tensorised["size"] = final_batch_tensorised[reference_id]["chr_lstm"].size(0)

    return final_batch_tensorised


class NERDataset(Dataset):

    def __init__(self,
                 mappings: dict = None,
                 instance_json_file: str = None,
                 testing: bool = None,
                 singleton_replacement_ratio: float = 0.0):

        self.mappings = mappings
        self.instance_json_file = instance_json_file
        self.testing = testing
        self.singleton_replacement_ratio = singleton_replacement_ratio

        self.singletons = set(self.extract_singletons())

        self.instances = dict()

        self.load_instances()

    def __len__(self):

        if self.testing:
            return len(self.instances) // 10
        else:
            return len(self.instances)

    def __getitem__(self, idx):

        return self.instances[idx]

    def extract_singletons(self):

        count = defaultdict(int)

        with open(os.path.abspath(self.instance_json_file), "r", encoding="UTF-8") as input_file:
            for line in input_file:
                if re.match("^$", line):
                    continue

                token = line.rstrip("\n").split("\t")[0]
                token = token.lower()
                count[token] += 1

        singletons = list()
        for token, token_count in count.items():
            if token_count == 1:
                singletons.append(token)

        return singletons

    def create_instance(self,
                        sequence_buffer: list = None):

        new_instance = {
            "tok": list(),
            "str": list(),

            "chr_cnn_literal": list(),
            "chr_cnn_utf8": list(),

            # "chr_lstm_type1": list(),
            # "chr_lstm_type2": list(),

            "lbl": list(),
        }

        for item in sequence_buffer:
            token_form = item[0]
            token_label = item[-1]

            token_lower = token_form.lower()
            token_encoded = token_form.encode("UTF-8")

            token_chr_cnn_literal = [self.mappings["characters_literal"].get("<bow>")] + \
                                    [self.mappings["characters_literal"].get(char) for char in token_form
                                     if self.mappings["characters_literal"].get(char)] + \
                                    [self.mappings["characters_literal"].get("<eow>")]

            # token_chr_lstm_type1 = [self.mappings["characters_type1"].get(char) for char in token_form
            #                         if self.mappings["characters_type1"].get(char)]

            token_chr_cnn_utf8 = [self.mappings["characters_utf8"].get("<bow>")] + \
                                 [char for char in token_encoded] + \
                                 [self.mappings["characters_utf8"].get("<eow>")]
            # token_chr_lstm_type2 = [char for char in token_encoded]

            if token_lower in self.singletons:
                if random.random() < self.singleton_replacement_ratio:
                    new_instance["tok"].append(
                        self.mappings["tokens"].get("<unk>")
                    )
                else:
                    new_instance["tok"].append(
                        self.mappings["tokens"].get(token_lower, self.mappings["tokens"].get("<unk>"))
                    )
            else:
                new_instance["tok"].append(
                    self.mappings["tokens"].get(token_lower, self.mappings["tokens"].get("<unk>"))
                )

            new_instance["str"].append(token_form)

            new_instance["chr_cnn_literal"].append(token_chr_cnn_literal)
            # new_instance["chr_lstm_type1"].append(token_chr_lstm_type1)

            new_instance["chr_cnn_utf8"].append(token_chr_cnn_utf8)
            # new_instance["chr_lstm_type2"].append(token_chr_lstm_type2)

            new_instance["lbl"].append(self.mappings["ner_labels"].get(token_label))

        return new_instance

    def load_instances(self):

        instance_idx = 0

        with open(os.path.abspath(self.instance_json_file), "r", encoding="UTF-8") as input_file:
            sequence_buffer = list()
            for line in input_file:
                if re.match("^$", line):
                    if len(sequence_buffer) > 0:
                        instance = self.create_instance(sequence_buffer=sequence_buffer)
                        self.instances[instance_idx] = instance
                        sequence_buffer = list()
                        instance_idx += 1

                    continue

                sequence_buffer.append(line.rstrip("\n").split("\t"))

            if len(sequence_buffer) > 0:
                instance = self.create_instance(sequence_buffer=sequence_buffer)
                self.instances[instance_idx] = instance
                instance_idx += 1


def collate_ner(batch,
                tok_pad_id: int = None,
                chr_pad_id_literal: int = None,
                chr_pad_id_utf8: int = None,
                options: dict = None):

    len_char_cnn_literal = list()
    len_char_cnn_utf8 = list()

    # len_chr_lstm_type1 = list()
    # len_chr_lstm_type2 = list()

    len_tok = list()

    for instance in batch:
        for item in instance["chr_cnn_literal"]:
            len_char_cnn_literal.append(len(item))

        for item in instance["chr_cnn_utf8"]:
            len_char_cnn_utf8.append(len(item))

        # for item in instance["chr_lstm_type1"]:
        #     len_chr_lstm_type1.append(len(item))

        # for item in instance["chr_lstm_type2"]:
        #     len_chr_lstm_type2.append(len(item))

        len_tok.append(len(instance["chr_cnn_literal"]))

    max_len_chr_cnn_literal = max(len_char_cnn_literal)
    max_len_chr_cnn_utf8 = max(len_char_cnn_utf8)

    # max_len_chr_lstm_type1 = max(len_chr_lstm_type1)
    # max_len_chr_lstm_type2 = max(len_chr_lstm_type2)

    max_len_tok = max(len_tok)

    if options.get("embeddings").get("chr_cnn").get("use"):
        max_kernel_size = max([k for k, f in options.get("embeddings").get("chr_cnn").get("cnn_filters")])

        if max_kernel_size > max_len_chr_cnn_literal:
            max_len_chr_cnn_literal = max_kernel_size

        if max_kernel_size > max_len_chr_cnn_utf8:
            max_len_chr_cnn_utf8 = max_kernel_size

    final_batch = {
        "chr_cnn_literal": list(),
        "chr_cnn_utf8": list(),

        # "chr_lstm_type1": list(),
        # "chr_len_literal": list(),

        # "chr_lstm_type2": list(),
        # "chr_len_utf8": list(),

        "tok": list(),
        "tok_len": list(),

        "str": list(),

        "elmo": None,

        "labels": list(),
        "mask": list(),
    }

    for instance in batch:
        # CHARS
        # =====

        chr_list_cnn_literal = list()
        # chr_list_lstm_type1 = list()
        # chr_list_len_type1 = list()

        chr_list_cnn_utf8 = list()
        # chr_list_lstm_type2 = list()
        # chr_list_len_type2 = list()

        mask = list()

        # TYPE1
        # ===========================================================
        # for token_chars in instance["chr_lstm_type1"]:
        #     cur_chars = copy.deepcopy(token_chars)
        #     chr_list_len_type1.append(len(cur_chars))
        #
        #     while len(cur_chars) < max_len_chr_lstm_type1:
        #         cur_chars.append(chr_pad_id_type1)
        #
        #     chr_list_lstm_type1.append(cur_chars)
        #
        #

        for token_chars in instance["chr_cnn_literal"]:
            cur_chars = copy.deepcopy(token_chars)

            while len(cur_chars) < max_len_chr_cnn_literal:
                cur_chars.append(chr_pad_id_literal)

            chr_list_cnn_literal.append(cur_chars)

            mask.append(1)

        while len(chr_list_cnn_literal) < max_len_tok:
            cur_chars = [chr_pad_id_literal for _ in range(max_len_chr_cnn_literal)]
            chr_list_cnn_literal.append(cur_chars)

            # cur_chars = [chr_pad_id_literal for _ in range(max_len_chr_lstm_type1)]
            # chr_list_lstm_type1.append(cur_chars)
            #
            # chr_list_len_type1.append(0)

            mask.append(0)

        # TYPE2
        # ===========================================================
        # for token_chars in instance["chr_lstm_type2"]:
        #     cur_chars = copy.deepcopy(token_chars)
        #     chr_list_len_type2.append(len(cur_chars))
        #
        #     while len(cur_chars) < max_len_chr_lstm_type2:
        #         cur_chars.append(chr_pad_id_utf8)
        #
        #     chr_list_lstm_type2.append(cur_chars)

        for token_chars in instance["chr_cnn_utf8"]:
            cur_chars = copy.deepcopy(token_chars)

            while len(cur_chars) < max_len_chr_cnn_utf8:
                cur_chars.append(chr_pad_id_utf8)

            chr_list_cnn_utf8.append(cur_chars)

        while len(chr_list_cnn_utf8) < max_len_tok:
            cur_chars = [chr_pad_id_utf8 for _ in range(max_len_chr_cnn_utf8)]
            chr_list_cnn_utf8.append(cur_chars)

            # cur_chars = [chr_pad_id_utf8 for _ in range(max_len_chr_lstm_type2)]
            # chr_list_lstm_type2.append(cur_chars)
            #
            # chr_list_len_type2.append(0)

        final_batch["chr_cnn_literal"].append(chr_list_cnn_literal)
        # final_batch["chr_lstm_type1"].append(chr_list_lstm_type1)
        # final_batch["chr_len_type1"].append(chr_list_len_type1)

        final_batch["chr_cnn_utf8"].append(chr_list_cnn_utf8)
        # final_batch["chr_lstm_type2"].append(chr_list_lstm_type2)
        # final_batch["chr_len_type2"].append(chr_list_len_type2)

        # STRINGS
        # =======

        final_batch["str"].append(instance["str"])

        # TOKENS
        # ======

        final_batch["tok_len"].append(len(instance["chr_cnn_literal"]))

        if options.get("embeddings").get("pretrained").get("use"):
            cur_tokens = copy.deepcopy(instance["tok"])

            while len(cur_tokens) < max_len_tok:
                cur_tokens.append(tok_pad_id)

            final_batch["tok"].append(cur_tokens)

        # LABELS
        # ======

        cur_labels = copy.deepcopy(instance["lbl"])

        while len(cur_labels) < max_len_tok:
            cur_labels.append(0)

        final_batch["labels"].append(cur_labels)
        final_batch["mask"].append(mask)

    final_batch["elmo"] = batch_to_ids(final_batch["str"])

    # final_batch["chr_lstm_type1"] = torch.LongTensor(final_batch["chr_lstm_type1"])
    final_batch["chr_cnn_literal"] = torch.LongTensor(final_batch["chr_cnn_literal"])
    # final_batch["chr_len_type1"] = torch.LongTensor(final_batch["chr_len_type1"])

    # final_batch["chr_lstm_type2"] = torch.LongTensor(final_batch["chr_lstm_type2"])
    final_batch["chr_cnn_utf8"] = torch.LongTensor(final_batch["chr_cnn_utf8"])
    # final_batch["chr_len_type2"] = torch.LongTensor(final_batch["chr_len_type2"])

    final_batch["tok"] = torch.LongTensor(final_batch["tok"])
    final_batch["tok_len"] = torch.LongTensor(final_batch["tok_len"])

    final_batch["labels"] = torch.LongTensor(final_batch["labels"])
    final_batch["mask"] = torch.LongTensor(final_batch["mask"])

    final_batch["size"] = final_batch["chr_cnn_literal"].size(0)

    return final_batch
