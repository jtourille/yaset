import copy
import os
import re

import torch
from allennlp.modules.elmo import batch_to_ids
from torch.utils.data import Dataset


class NERDataset(Dataset):

    def __init__(self,
                 mappings: dict = None,
                 instance_json_file: str = None,
                 testing: bool = None):

        self.mappings = mappings
        self.instance_json_file = instance_json_file
        self.testing = testing

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

        new_instance = {
            "tok": list(),
            "str": list(),
            "chr": list(),

            "lbl": list(),
        }

        for item in sequence_buffer:
            token_form = item[0]
            token_label = item[-1]

            token_lower = token_form.lower()
            token_chars = [self.mappings["characters"].get("<bow>")] + \
                          [self.mappings["characters"].get(char) for char in token_form
                           if self.mappings["characters"].get(char)] + \
                          [self.mappings["characters"].get("<eow>")]

            new_instance["tok"].append(
                self.mappings["tokens"].get(token_lower, self.mappings["tokens"].get("<unk>"))
            )
            new_instance["str"].append(token_form)
            new_instance["chr"].append(token_chars)
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
                chr_pad_id: int = None,
                options: dict = None):

    len_chr = list()
    len_tok = list()

    for instance in batch:
        for item in instance["chr"]:
            len_chr.append(len(item))

        len_tok.append(len(instance["chr"]))

    max_len_chr = max(len_chr)
    max_len_tok = max(len_tok)

    final_batch = {
        "chr": list(),
        "chr_len": list(),

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

        chr_list = list()
        chr_list_len = list()
        mask = list()

        for token_chars in instance["chr"]:
            cur_chars = copy.deepcopy(token_chars)
            chr_list_len.append(len(cur_chars))

            while len(cur_chars) < max_len_chr:
                cur_chars.append(chr_pad_id)

            chr_list.append(cur_chars)

            mask.append(1)

        while len(chr_list) < max_len_tok:
            cur_chars = [chr_pad_id for _ in range(max_len_chr)]
            chr_list.append(cur_chars)
            chr_list_len.append(0)

            mask.append(0)

        final_batch["chr"].append(chr_list)
        final_batch["chr_len"].append(chr_list_len)

        # STRINGS
        # =======

        final_batch["str"].append(instance["str"])

        # TOKENS
        # ======

        final_batch["tok_len"].append(len(instance["chr"]))

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

    final_batch["chr"] = torch.LongTensor(final_batch["chr"])
    final_batch["chr_len"] = torch.LongTensor(final_batch["chr_len"])
    final_batch["tok"] = torch.LongTensor(final_batch["tok"])
    final_batch["tok_len"] = torch.LongTensor(final_batch["tok_len"])

    final_batch["labels"] = torch.LongTensor(final_batch["labels"])
    final_batch["mask"] = torch.LongTensor(final_batch["mask"])

    final_batch["size"] = final_batch["chr"].size(0)

    return final_batch
