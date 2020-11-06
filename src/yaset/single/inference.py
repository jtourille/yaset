import copy
import os

import numpy as np
import torch
import torch.nn as nn
from allennlp.modules.elmo import batch_to_ids
from transformers.tokenization_bert import BertTokenizer
from yaset.utils.misc import flatten


class NERModel:
    def __init__(
        self,
        mappings: dict = None,
        model: nn.Module = None,
        options: dict = None,
        model_dir: str = None,
    ):

        self.mappings = mappings
        self.model = model
        self.options = options

        self.bert_use = options.get("embeddings").get("bert").get("use")
        self.bert_voc_dir = os.path.join(
            model_dir, "embeddings", "bert", "vocab"
        )
        self.bert_lowercase = (
            options.get("embeddings").get("bert").get("do_lower_case")
        )

        self.pretrained_use = (
            options.get("embeddings").get("pretrained").get("use")
        )
        self.elmo_use = options.get("embeddings").get("pretrained").get("use")
        self.char_use = options.get("embeddings").get("chr_cnn").get("use")

        if self.bert_use:
            self.bert_tokenizer = BertTokenizer.from_pretrained(
                self.bert_voc_dir, do_lower_case=self.bert_lowercase
            )

        self.tok_pad_id = None
        self.chr_pad_id_literal = None
        self.chr_pad_id_utf8 = None

        if self.pretrained_use:
            self.tok_pad_id = mappings.get("toks").get("pad_id")

        if self.char_use:
            self.chr_pad_id_literal = (
                mappings.get("chrs").get("char_literal").get("<pad>")
            )
            self.chr_pad_id_utf8 = (
                mappings.get("chrs").get("char_utf8").get("<pad>")
            )

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

        new_instance = dict()

        if self.char_use:
            new_instance["chr_cnn_literal"] = list()
            new_instance["chr_cnn_utf8"] = list()

        if self.pretrained_use:
            new_instance["tok"] = list()

        instance_str = list()

        for token in sentence:
            token_form = token
            token_lower = token.lower()
            instance_str.append(token)

            if self.char_use:
                token_lower = token_form.lower()
                token_encoded = token_form.encode("UTF-8")

                token_chr_cnn_literal = (
                    [
                        self.mappings.get("chrs")
                        .get("char_literal")
                        .get("<bow>")
                    ]
                    + [
                        self.mappings.get("chrs").get("char_literal").get(char)
                        for char in token_form
                        if self.mappings.get("chrs")
                        .get("char_literal")
                        .get(char)
                    ]
                    + [
                        self.mappings.get("chrs")
                        .get("char_literal")
                        .get("<eow>")
                    ]
                )

                token_chr_cnn_utf8 = (
                    [self.mappings.get("chrs").get("char_utf8").get("<bow>")]
                    + [char for char in token_encoded]
                    + [self.mappings.get("chrs").get("char_utf8").get("<eow>")]
                )

                new_instance["chr_cnn_literal"].append(token_chr_cnn_literal)
                new_instance["chr_cnn_utf8"].append(token_chr_cnn_utf8)

            if self.pretrained_use:
                new_instance["tok"].append(
                    self.mappings.get("toks")
                    .get("tokens")
                    .get(
                        token_lower,
                        self.mappings.get("toks").get("oov_id"),
                    )
                )

            new_instance["str"] = instance_str

            if self.bert_use:
                seq_token_pieces = list(
                    map(self.bert_tokenizer.tokenize, instance_str)
                )
                seq_token_lens = list(map(len, seq_token_pieces))
                seq_token_pieces = (
                    ["[CLS]"] + list(flatten(seq_token_pieces)) + ["[SEP]"]
                )
                seq_token_start = 1 + np.cumsum([0] + seq_token_lens[:-1])
                seq_input_ids = self.bert_tokenizer.convert_tokens_to_ids(
                    seq_token_pieces
                )

                new_instance["bert_input_ids"] = seq_input_ids
                new_instance["bert_seq_token_start"] = seq_token_start

        return new_instance

    def collate_sentences(self, batch):

        len_char_cnn_literal = list()
        len_char_cnn_utf8 = list()

        len_tok = list()

        for instance in batch:
            if self.char_use:
                for item in instance["chr_cnn_literal"]:
                    len_char_cnn_literal.append(len(item))

                for item in instance["chr_cnn_utf8"]:
                    len_char_cnn_utf8.append(len(item))

            len_tok.append(len(instance["str"]))

        max_len_tok = max(len_tok)

        if self.char_use:
            max_len_chr_cnn_literal = max(len_char_cnn_literal)
            max_len_chr_cnn_utf8 = max(len_char_cnn_utf8)

            max_kernel_size = max(
                [
                    k
                    for k, f in self.options.get("embeddings")
                    .get("chr_cnn")
                    .get("cnn_filters")
                ]
            )

            if max_kernel_size > max_len_chr_cnn_literal:
                max_len_chr_cnn_literal = max_kernel_size

            if max_kernel_size > max_len_chr_cnn_utf8:
                max_len_chr_cnn_utf8 = max_kernel_size

        final_batch = {
            "chr_cnn_literal": list(),
            "chr_cnn_utf8": list(),
            "tok": list(),
            "tok_len": list(),
            "str": list(),
            "elmo": None,
            "mask": list(),
        }

        all_token_start = list()
        all_input_ids = list()

        for instance in batch:
            # MASK
            # ===========================================================
            mask = list()

            for _ in instance["str"]:
                mask.append(1)

            while len(mask) < max_len_tok:
                mask.append(0)

            if self.char_use:
                # CHAR LITERAL
                # =======================================================
                chr_list_cnn_literal = list()

                for token_chars in instance["chr_cnn_literal"]:
                    cur_chars = copy.deepcopy(token_chars)

                    while len(cur_chars) < max_len_chr_cnn_literal:
                        cur_chars.append(self.chr_pad_id_literal)

                    chr_list_cnn_literal.append(cur_chars)

                    # mask.append(1)

                while len(chr_list_cnn_literal) < max_len_tok:
                    cur_chars = [
                        self.chr_pad_id_literal
                        for _ in range(max_len_chr_cnn_literal)
                    ]
                    chr_list_cnn_literal.append(cur_chars)

                    # mask.append(0)

                # CHAR UTF-8
                # ===========================================================
                chr_list_cnn_utf8 = list()

                for token_chars in instance["chr_cnn_utf8"]:
                    cur_chars = copy.deepcopy(token_chars)

                    while len(cur_chars) < max_len_chr_cnn_utf8:
                        cur_chars.append(self.chr_pad_id_utf8)

                    chr_list_cnn_utf8.append(cur_chars)

                while len(chr_list_cnn_utf8) < max_len_tok:
                    cur_chars = [
                        self.chr_pad_id_utf8
                        for _ in range(max_len_chr_cnn_utf8)
                    ]
                    chr_list_cnn_utf8.append(cur_chars)

                final_batch["chr_cnn_literal"].append(chr_list_cnn_literal)
                final_batch["chr_cnn_utf8"].append(chr_list_cnn_utf8)

            # STRINGS
            # =======
            final_batch["str"].append(instance["str"])

            final_batch["tok_len"].append(len(instance["str"]))

            if self.pretrained_use:
                # TOKENS
                # ======
                cur_tokens = copy.deepcopy(instance["tok"])

                while len(cur_tokens) < max_len_tok:
                    cur_tokens.append(self.tok_pad_id)

                final_batch["tok"].append(cur_tokens)

            final_batch["mask"].append(mask)

            if self.bert_use:
                # BERT
                # ====
                all_token_start.append(instance["bert_seq_token_start"])
                all_input_ids.append(instance["bert_input_ids"])

        if self.bert_use:
            # BERT BATCHING
            # =============
            max_seq_len_pieces = max([len(seq) for seq in all_input_ids])
            max_seq_len_str = max([len(seq) for seq in final_batch["str"]])

            tensor_all_input_ids = list()
            tensor_all_input_type_ids = list()
            tensor_all_input_mask = list()
            tensor_all_indices = list()

            for input_ids, seq_indices in zip(all_input_ids, all_token_start):
                input_type_ids = [0] * len(input_ids)
                input_mask = [1] * len(input_ids)

                while len(input_ids) < max_seq_len_pieces:
                    input_ids.append(0)
                    input_type_ids.append(0)
                    input_mask.append(0)

                tensor_all_input_ids.append(torch.LongTensor([input_ids]))
                tensor_all_input_type_ids.append(
                    torch.LongTensor([input_type_ids])
                )
                tensor_all_input_mask.append(torch.LongTensor([input_mask]))
                tensor_all_indices.append(torch.LongTensor(seq_indices))

            tensor_all_input_ids = torch.cat(tensor_all_input_ids, 0)
            tensor_all_input_type_ids = torch.cat(tensor_all_input_type_ids, 0)
            tensor_all_input_mask = torch.cat(tensor_all_input_mask, 0)

            final_batch["bert_all_input_ids"] = tensor_all_input_ids
            final_batch["bert_all_input_type_ids"] = tensor_all_input_type_ids
            final_batch["bert_all_input_mask"] = tensor_all_input_mask
            final_batch["bert_all_indices"] = tensor_all_indices
            final_batch["bert_max_seq_len_str"] = max_seq_len_str

        final_batch["elmo"] = batch_to_ids(final_batch["str"])

        final_batch["chr_cnn_literal"] = torch.LongTensor(
            final_batch["chr_cnn_literal"]
        )
        final_batch["chr_cnn_utf8"] = torch.LongTensor(
            final_batch["chr_cnn_utf8"]
        )

        final_batch["tok"] = torch.LongTensor(final_batch["tok"])
        final_batch["tok_len"] = torch.LongTensor(final_batch["tok_len"])

        final_batch["mask"] = torch.LongTensor(final_batch["mask"])

        final_batch["size"] = final_batch["chr_cnn_literal"].size(0)

        return final_batch
