import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo
from transformers.modeling_bert import BertModel, BertConfig
from transformers.tokenization_bert import BertTokenizer

from yaset.nn.cnn import CharCNN
from yaset.utils.misc import flatten


class Embedder(nn.Module):

    def __init__(self, embeddings_options: dict = None,
                 pretrained_matrix: np.ndarray = None,
                 pretrained_matrix_size: (int, int) = None,
                 mappings: dict = None,
                 embedding_root_dir: str = None):
        super().__init__()

        self.embeddings_options = embeddings_options
        self.pretrained_embedding = None
        self.pretrained_matrix_size = pretrained_matrix_size
        self.mappings = mappings

        self.char_cnn_embedding = None
        self.char_cnn_type = None
        self.elmo_embedding = None
        self.bert_embedding = None
        self.pos_embedding = None

        self.embedding_size = 0

        if self.embeddings_options.get("pretrained").get("use"):
            logging.debug("Initializing pretrained_embeddings module")
            self.pretrained_embedding = nn.Embedding(self.pretrained_matrix_size[0], self.pretrained_matrix_size[1])
            self.embedding_size += self.pretrained_embedding.weight.size(1)

            if pretrained_matrix is not None:
                self.pretrained_embedding.from_pretrained(torch.from_numpy(pretrained_matrix), freeze=False)

        if self.embeddings_options.get("chr_cnn").get("use"):
            logging.debug("Initializing char_cnn module")

            if self.embeddings_options.get("chr_cnn").get("type") == "literal":
                embed_len = len(mappings["characters_literal"])
                self.char_cnn_type = "literal"

            elif self.embeddings_options.get("chr_cnn").get("type") == "utf8":
                embed_len = 259
                self.char_cnn_type = "utf8"

            else:
                logging.info(self.embeddings_options)
                raise Exception("Unrecognized character type: {}".format(
                    self.embeddings_options.get("chr_cnn").get("type")

                ))

            self.char_cnn_embedding = nn.Embedding(embed_len,
                                                   self.embeddings_options.get("chr_cnn").get("char_embedding_size"))
            self.char_cnn = CharCNN(char_embedding=self.char_cnn_embedding,
                                    filters=self.embeddings_options.get("chr_cnn").get("cnn_filters"))

            for kernel_size, num_filters in self.embeddings_options.get("chr_cnn").get("cnn_filters"):
                self.embedding_size += num_filters

            torch.nn.init.xavier_uniform_(self.char_cnn_embedding.weight)

        if self.embeddings_options.get("elmo").get("use"):
            logging.debug("Initializing elmo_embeddings module")
            weight_file = os.path.join(embedding_root_dir, "elmo",
                                       os.path.basename(self.embeddings_options.get("elmo").get("weight_path")))
            options_file = os.path.join(embedding_root_dir, "elmo",
                                        os.path.basename(self.embeddings_options.get("elmo").get("options_path")))

            self.elmo_embedding = Elmo(options_file,
                                       weight_file, 1, dropout=0.0,
                                       requires_grad=self.embeddings_options.get("elmo").get("fine_tune"))
            self.embedding_size += 1024

        if self.embeddings_options.get("bert").get("use"):
            logging.debug("Initializing bert_embeddings module")
            model_file = os.path.join(embedding_root_dir, "bert",
                                      os.path.basename(self.embeddings_options.get("bert").get("model_file")))
            vocab_dir = os.path.join(embedding_root_dir, "bert", "vocab")
            config_file = os.path.join(embedding_root_dir, "bert",
                                       os.path.basename(self.embeddings_options.get("bert").get("config_file")))

            self.bert_embedding = BertEmbeddings(
                model_type=self.embeddings_options.get("bert").get("type"),
                model_file=model_file,
                model_config_file=config_file,
                do_lower_case=self.embeddings_options.get("bert").get("do_lower_case"),
                vocab_dir=vocab_dir,
                fine_tune=self.embeddings_options.get("bert").get("fine_tune"),
                only_final_layer=embeddings_options.get("bert").get("only_final_layer")
            )
            self.embedding_size += self.bert_embedding.hidden_size

    def forward(self, batch, cuda):

        to_concat = list()

        if self.char_cnn_embedding:
            if self.char_cnn_type == "literal":
                char_embed = self.char_cnn_embedding(batch["chr_cnn_literal"])
            else:
                char_embed = self.char_cnn_embedding(batch["chr_cnn_utf8"])

            cnn_output = self.char_cnn(char_embed)
            to_concat.append(cnn_output)

        if self.pretrained_embedding:
            token_embed = self.pretrained_embedding(batch["tok"])
            to_concat.append(token_embed)

        if self.elmo_embedding:
            elmo_embed = self.elmo_embedding(batch["elmo"])
            to_concat.append(elmo_embed["elmo_representations"][0])

        if self.bert_embedding:
            bert_embed = self.bert_embedding(batch["str"], cuda)
            to_concat.append(bert_embed)

        final_output = torch.cat(to_concat, 2)

        return final_output


class BertEmbeddings(nn.Module):

    def __init__(self,
                 model_config_file: str = None,
                 model_file: str = None,
                 model_type: str = None,
                 do_lower_case: bool = None,
                 vocab_dir: str = None,
                 fine_tune: bool = False,
                 only_final_layer: bool = False):
        super().__init__()

        self.model_config_file = model_config_file
        self.model_file = model_file
        self.model_type = model_type
        self.do_lower_case = do_lower_case
        self.vocab_dir = vocab_dir
        self.fine_tune = fine_tune
        self.only_final_layer = only_final_layer

        self.tokenizer = BertTokenizer.from_pretrained(self.vocab_dir, do_lower_case=self.do_lower_case)
        self.config = BertConfig.from_pretrained(self.model_config_file)

        self.config.output_hidden_states = True
        self.config.output_attentions = False

        if self.model_type == "tensorflow":
            self.model = BertModel.from_pretrained(self.model_file, from_tf=True)

        elif self.model_type == "pytorch":
            self.model = BertModel.from_pretrained(self.model_file, config=self.config)

        else:
            raise Exception

        self.weights = nn.Parameter(torch.FloatTensor([0.0] * self.model.config.num_hidden_layers), requires_grad=True)
        self.gamma = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        self.hidden_size = self.model.config.hidden_size

    def forward(self, sequence_list, cuda):

        if self.fine_tune:
            selected_encoders_layers = self.compute_embeddings(sequence_list, cuda)

        else:
            with torch.no_grad():
                self.model.eval()
                selected_encoders_layers = self.compute_embeddings(sequence_list, cuda)

        if self.only_final_layer:
            return selected_encoders_layers[-1]

        else:
            softmaxed_weights = F.softmax(self.weights, dim=0)
            weighted_sum = torch.mul(selected_encoders_layers[0], softmaxed_weights[0])

            for i, layer in enumerate(selected_encoders_layers[1:], start=1):
                weighted_sum = torch.add(weighted_sum, torch.mul(selected_encoders_layers[i], softmaxed_weights[i]))

            scaled_weighted_sum = torch.mul(weighted_sum, self.gamma)

            return scaled_weighted_sum

    def compute_embeddings(self, sequence_list, cuda):

        all_token_pieces = list()
        all_token_start = list()

        for sequence in sequence_list:

            seq_token_pieces = list(map(self.tokenizer.tokenize, sequence))
            seq_token_lens = list(map(len, seq_token_pieces))
            seq_token_pieces = ["[CLS]"] + list(flatten(seq_token_pieces)) + ["[SEP]"]
            seq_token_start = 1 + np.cumsum([0] + seq_token_lens[:-1])

            all_token_pieces.append(seq_token_pieces)
            all_token_start.append(seq_token_start)

        max_seq_len_pieces = max([len(seq) for seq in all_token_pieces])
        max_seq_len_str = max([len(seq) for seq in sequence_list])

        tensor_all_input_ids = list()
        tensor_all_input_type_ids = list()
        tensor_all_input_mask = list()
        tensor_all_indices = list()

        for seq_token_pieces, seq_indices in zip(all_token_pieces, all_token_start):
            input_ids = self.tokenizer.convert_tokens_to_ids(seq_token_pieces)
            input_type_ids = [0] * len(input_ids)
            input_mask = [1] * len(input_ids)

            while len(input_ids) < max_seq_len_pieces:
                input_ids.append(0)
                input_type_ids.append(0)
                input_mask.append(0)

            tensor_all_input_ids.append(torch.LongTensor([input_ids]))
            tensor_all_input_type_ids.append(torch.LongTensor([input_type_ids]))
            tensor_all_input_mask.append(torch.LongTensor([input_mask]))
            tensor_all_indices.append(torch.LongTensor(seq_indices))

        tensor_all_input_ids = torch.cat(tensor_all_input_ids, 0)
        tensor_all_input_type_ids = torch.cat(tensor_all_input_type_ids, 0)
        tensor_all_input_mask = torch.cat(tensor_all_input_mask, 0)

        if cuda:
            tensor_all_input_ids = tensor_all_input_ids.cuda()
            tensor_all_input_type_ids = tensor_all_input_type_ids.cuda()
            tensor_all_input_mask = tensor_all_input_mask.cuda()
            tensor_all_indices = [item.cuda() for item in tensor_all_indices]

        last_hidden_state, pooler_output, all_encoders_layers = self.model(
            tensor_all_input_ids,
            token_type_ids=tensor_all_input_type_ids,
            attention_mask=tensor_all_input_mask
        )

        selected_encoders_layers = list()
        if self.only_final_layer:
            selected_layer = list()
            for a, i in zip(all_encoders_layers[-1], tensor_all_indices):
                selected = torch.index_select(a, 0, i)

                if selected.size(0) < max_seq_len_str:
                    padding = torch.zeros(max_seq_len_str - selected.size(0), selected.size(1))
                    if cuda:
                        padding = padding.cuda()
                    selected = torch.cat([selected, padding], 0)

                selected_layer.append(selected.unsqueeze(0))
            selected_layer = torch.cat(selected_layer, 0)
            selected_encoders_layers.append(selected_layer)

        else:
            for layer in all_encoders_layers[1:]:
                selected_layer = list()
                for a, i in zip(layer, tensor_all_indices):
                    selected = torch.index_select(a, 0, i)

                    if selected.size(0) < max_seq_len_str:
                        padding = torch.zeros(max_seq_len_str - selected.size(0), selected.size(1))
                        if cuda:
                            padding = padding.cuda()
                        selected = torch.cat([selected, padding], 0)

                    selected_layer.append(selected.unsqueeze(0))
                selected_layer = torch.cat(selected_layer, 0)
                selected_encoders_layers.append(selected_layer)

        return selected_encoders_layers
