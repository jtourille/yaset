import copy

import torch
import torch.nn as nn
from allennlp.modules.elmo import batch_to_ids


class NERModel:
    def __init__(
        self,
        mappings: dict = None,
        model: nn.Module = None,
        options: dict = None,
    ):

        self.mappings = mappings
        self.model = model

        self.tok_pad_id = mappings["tokens"].get("<pad>")
        self.chr_pad_id = mappings["characters"].get("<pad>")
        self.options = options

    def __call__(self, sentences, cuda, *args, **kwargs):

        sentence_ids = list()
        for sentence in sentences:
            sentence_ids.append(self.sentence_to_ids(sentence))

        batch = self.collate_sentences(sentence_ids)

        self.model.eval()
        with torch.no_grad():
            labels = self.model.infer_labels(batch, cuda)

        return labels

    def dev_predict(self, sentences, cuda, *arg, **kwargs):

        sentence_ids = list()
        for sentence in sentences:
            sentence_ids.append(self.sentence_to_ids(sentence))

        batch = self.collate_sentences(sentence_ids)

        with torch.no_grad():
            labels = self.model.dev_infer_labels(batch, cuda)

        return labels

    def sentence_to_ids(self, sentence):

        new_instance = {
            "tok": list(),
            "str": list(),
            "chr_cnn": list(),
            "chr_lstm": list(),
        }

        for token in sentence:
            token_lower = token.lower()
            token_chr_cnn = (
                [self.mappings["characters"].get("<bow>")]
                + [
                    self.mappings["characters"].get(char)
                    for char in token
                    if self.mappings["characters"].get(char)
                ]
                + [self.mappings["characters"].get("<eow>")]
            )
            token_chr_lstm = [
                self.mappings["characters"].get(char)
                for char in token
                if self.mappings["characters"].get(char)
            ]

            new_instance["tok"].append(
                self.mappings["tokens"].get(
                    token_lower, self.mappings["tokens"].get("<unk>")
                )
            )
            new_instance["str"].append(token)
            new_instance["chr_cnn"].append(token_chr_cnn)
            new_instance["chr_lstm"].append(token_chr_lstm)

        return new_instance

    def collate_sentences(self, sentences):

        len_chr_lstm = list()
        len_chr_cnn = list()
        len_tok = list()

        for sentence in sentences:
            for item in sentence["chr_lstm"]:
                len_chr_lstm.append(len(item))

            for item in sentence["chr_cnn"]:
                len_chr_cnn.append(len(item))

            len_tok.append(len(sentence["chr_lstm"]))

        max_len_chr_lstm = max(len_chr_lstm)
        max_len_chr_cnn = max(len_chr_cnn)
        max_len_tok = max(len_tok)

        final_batch = {
            "chr_cnn": list(),
            "chr_lstm": list(),
            "chr_len": list(),
            "tok": list(),
            "tok_len": list(),
            "str": list(),
            "elmo": None,
            "mask": list(),
        }

        for sentence in sentences:
            # CHARS
            # =====

            chr_list_cnn = list()
            chr_list_lstm = list()
            chr_list_len = list()
            mask = list()

            for token_chars in sentence["chr_lstm"]:
                cur_chars = copy.deepcopy(token_chars)
                chr_list_len.append(len(cur_chars))

                while len(cur_chars) < max_len_chr_lstm:
                    cur_chars.append(self.chr_pad_id)

                chr_list_lstm.append(cur_chars)

                mask.append(1)

            for token_chars in sentence["chr_cnn"]:
                cur_chars = copy.deepcopy(token_chars)

                while len(cur_chars) < max_len_chr_cnn:
                    cur_chars.append(self.chr_pad_id)

                chr_list_cnn.append(cur_chars)

            while len(chr_list_cnn) < max_len_tok:
                cur_chars = [self.chr_pad_id for _ in range(max_len_chr_cnn)]
                chr_list_cnn.append(cur_chars)

                cur_chars = [self.chr_pad_id for _ in range(max_len_chr_lstm)]
                chr_list_lstm.append(cur_chars)

                chr_list_len.append(0)

                mask.append(0)

            final_batch["chr_cnn"].append(chr_list_cnn)
            final_batch["chr_lstm"].append(chr_list_lstm)
            final_batch["chr_len"].append(chr_list_len)

            # STRINGS
            # =======

            final_batch["str"].append(sentence["str"])

            # TOKENS
            # ======

            final_batch["tok_len"].append(len(sentence["chr_lstm"]))

            if self.options.get("embeddings").get("pretrained").get("use"):
                cur_tokens = copy.deepcopy(sentence["tok"])

                while len(cur_tokens) < max_len_tok:
                    cur_tokens.append(self.tok_pad_id)

                final_batch["tok"].append(cur_tokens)

            final_batch["mask"].append(mask)

        final_batch["elmo"] = batch_to_ids(final_batch["str"])

        final_batch["chr_lstm"] = torch.LongTensor(final_batch["chr_lstm"])
        final_batch["chr_cnn"] = torch.LongTensor(final_batch["chr_cnn"])
        final_batch["chr_len"] = torch.LongTensor(final_batch["chr_len"])

        final_batch["tok"] = torch.LongTensor(final_batch["tok"])
        final_batch["tok_len"] = torch.LongTensor(final_batch["tok_len"])

        final_batch["mask"] = torch.LongTensor(final_batch["mask"])

        final_batch["size"] = final_batch["chr_lstm"].size(0)

        return final_batch
