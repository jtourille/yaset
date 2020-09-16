import copy
from collections import defaultdict

import torch
import torch.nn as nn
from allennlp.modules.elmo import batch_to_ids


class NERModel:
    def __init__(
        self,
        model_mappings: dict = None,
        model: nn.Module = None,
        model_options: dict = None,
        sentence_size_mapping: dict = None,
    ):

        self.mappings = model_mappings
        self.model = model
        self.options = model_options
        self.sentence_size_mapping = sentence_size_mapping

    def __call__(self, sentences, cuda, *args, **kwargs):

        sentence_ids = list()
        for sentence in sentences:
            sentence_ids.append(self.sentence_to_ids(sentence))

        batch = self.collate_sentences(sentence_ids)

        self.model.eval()
        with torch.no_grad():
            labels = self.model.infer_labels(batch, cuda)

        return labels

    def sentence_to_ids(self, sentence):

        new_instance = defaultdict(dict)

        for model_id, model_mapping in self.mappings.items():
            model_instance = {
                "tok": list(),
                "str": list(),
                "chr_cnn": list(),
                "chr_lstm": list(),
            }

            for token in sentence:
                token_form = token

                token_lower = token_form.lower()
                token_chr_cnn = (
                    [model_mapping["characters"].get("<bow>")]
                    + [
                        model_mapping["characters"].get(char)
                        for char in token_form
                        if model_mapping["characters"].get(char)
                    ]
                    + [model_mapping["characters"].get("<eow>")]
                )
                token_chr_lstm = [
                    model_mapping["characters"].get(char)
                    for char in token_form
                    if model_mapping["characters"].get(char)
                ]

                model_instance["tok"].append(
                    model_mapping["tokens"].get(
                        token_lower, model_mapping["tokens"].get("<unk>")
                    )
                )
                model_instance["str"].append(token_form)
                model_instance["chr_cnn"].append(token_chr_cnn)
                model_instance["chr_lstm"].append(token_chr_lstm)

            for (lower, upper), sent_id in self.sentence_size_mapping.items():
                if lower <= len(model_instance["str"]) <= upper:
                    model_instance["sent_size"] = sent_id
                    break

            else:
                raise Exception("Problem")

            new_instance[model_id] = model_instance

        return new_instance

    def collate_sentences(self, sentences):

        len_chr_lstm = list()
        len_char_cnn = list()
        len_tok = list()

        for instance in sentences:
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

        for instance in sentences:
            for model_id, model_instance in instance.items():

                tok_pad_id = self.mappings[model_id].get("tokens").get("<pad>")
                chr_pad_id = (
                    self.mappings[model_id].get("characters").get("<pad>")
                )

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

                final_batch[model_id]["tok_len"].append(
                    len(model_instance["chr_lstm"])
                )

                if (
                    self.options.get(model_id)
                    .get("embeddings")
                    .get("pretrained")
                    .get("use")
                ):
                    cur_tokens = copy.deepcopy(model_instance["tok"])

                    while len(cur_tokens) < max_len_tok:
                        cur_tokens.append(tok_pad_id)

                    final_batch[model_id]["tok"].append(cur_tokens)

                final_batch[model_id]["mask"].append(mask)

                # SENTENCE SIZE
                # =============

                final_batch[model_id]["sent_size"].append(
                    model_instance["sent_size"]
                )

        final_batch_tensorised = dict()
        for model_id, model_batch in final_batch.items():
            final_batch_tensorised[model_id] = dict()

            final_batch_tensorised[model_id]["str"] = model_batch["str"]
            final_batch_tensorised[model_id]["elmo"] = batch_to_ids(
                model_batch["str"]
            )

            final_batch_tensorised[model_id]["chr_lstm"] = torch.LongTensor(
                model_batch["chr_lstm"]
            )
            final_batch_tensorised[model_id]["chr_cnn"] = torch.LongTensor(
                model_batch["chr_cnn"]
            )
            final_batch_tensorised[model_id]["chr_len"] = torch.LongTensor(
                model_batch["chr_len"]
            )

            final_batch_tensorised[model_id]["tok"] = torch.LongTensor(
                model_batch["tok"]
            )
            final_batch_tensorised[model_id]["tok_len"] = torch.LongTensor(
                model_batch["tok_len"]
            )

            final_batch_tensorised[model_id]["mask"] = torch.LongTensor(
                model_batch["mask"]
            )

            final_batch_tensorised[model_id]["size"] = final_batch_tensorised[
                model_id
            ]["chr_lstm"].size(0)

        final_batch_tensorised["sent_size"] = torch.LongTensor(
            final_batch["model_1"]["sent_size"]
        )
        final_batch_tensorised["mask"] = torch.LongTensor(
            final_batch["model_1"]["mask"]
        )
        final_batch_tensorised["size"] = final_batch_tensorised["model_1"][
            "chr_lstm"
        ].size(0)

        return final_batch_tensorised
