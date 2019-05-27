import logging

import numpy as np
import torch
import torch.functional as F
import torch.nn as nn
from allennlp.modules.elmo import Elmo
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer

from .cnn import CharCNN


class Embedder(nn.Module):

    def __init__(self, embeddings_options: dict = None,
                 pretrained_matrix: np.ndarray = None,
                 mappings: dict = None):
        super().__init__()

        self.embeddings_options = embeddings_options
        self.pretrained_embedding = None
        self.mappings = mappings

        self.char_embedding = None
        self.elmo_embedding = None
        self.bert_embedding = None
        self.pos_embedding = None

        self.embedding_size = 0

        if self.embeddings_options.get("pretrained").get("use"):
            if pretrained_matrix is not None:
                logging.debug("Creating embedding object and loading pretrained matrix")
                self.pretrained_embedding = nn.Embedding(pretrained_matrix.shape[0], pretrained_matrix.shape[1])
                self.pretrained_embedding.weight.data.copy_(torch.from_numpy(pretrained_matrix))
                self.embedding_size += self.pretrained_embedding.weight.size(1)
            else:
                raise Exception("The pretrained matrix object is None")

        if self.embeddings_options.get("characters").get("use"):
            self.char_embedding = nn.Embedding(len(mappings["characters"]),
                                               self.embeddings_options.get("characters").get("char_embedding_size"))
            self.char_cnn = CharCNN(char_embedding=self.char_embedding,
                                    filters=self.embeddings_options.get("characters").get("cnn_filters"))

            for kernel_size, num_filters in self.embeddings_options.get("characters").get("cnn_filters"):
                self.embedding_size += num_filters

            torch.nn.init.xavier_uniform_(self.char_embedding.weight)

        if self.embeddings_options.get("elmo").get("use"):
            self.elmo_embedding = Elmo(self.embeddings_options.get("elmo").get("options_path"),
                                       self.embeddings_options.get("elmo").get("weight_path"), 1, dropout=0.0,
                                       requires_grad=False)
            self.embedding_size += 1024

        if self.embeddings_options.get("bert").get("use"):
            self.bert_embedding = BertEmbeddings(
                model_name=self.embeddings_options.get("bert").get("model_path"),
                do_lower_case=self.embeddings_options.get("bert").get("do_lower_case"),
                vocab_file=self.embeddings_options.get("bert").get("vocab_file"))
            self.embedding_size += 768

    def forward(self, batch, cuda):

        to_concat = list()

        if self.char_embedding:
            char_embed = self.char_embedding(batch["chr"])
            cnn_output = self.char_cnn(char_embed)
            to_concat.append(cnn_output)

        if self.pretrained_embedding:
            token_embed = self.pretrained_embedding(batch["tok"])
            to_concat.append(token_embed)

        if self.elmo_embedding:
            elmo_embed = self.elmo_embedding(batch["elmo"])
            to_concat.append(elmo_embed["elmo_representations"][0])

        if self.bert_embedding:
            bert_embed = self.bert_embedding.encode(batch["str"], is_tokenized=True)
            bert_embed = bert_embed[:, 1:-1, :]
            bert_embed = torch.FloatTensor(bert_embed)
            if cuda:
                bert_embed = bert_embed.cuda()
            bert_embed = bert_embed.view(-1, 1024)
            bert_embed = self.bert_projection(bert_embed)
            bert_embed = bert_embed.view(batch["size"], -1, self.bert_projection_size)
            to_concat.append(bert_embed)

        final_output = torch.cat(to_concat, 2)

        return final_output


class BertEmbeddings(nn.Module):

    def __init__(self, model_name: str = None, do_lower_case: bool = None,
                 vocab_file: str = None):
        super().__init__()

        self.model_name = model_name
        self.do_lower_case = do_lower_case
        self.vocab_file = vocab_file
        self.weights = nn.Parameter(torch.randn(12), requires_grad=True)

        self.tokenizer = BertTokenizer.from_pretrained(self.vocab_file, do_lower_case=self.do_lower_case)
        self.model = BertModel.from_pretrained(self.model_name)

    def forward(self, sequence_list, cuda):

        with torch.no_grad():
            all_token_pieces = list()
            all_indices = list()

            for sequence in sequence_list:
                seq_token_pieces = list()
                seq_indices = list()
                seq_index = 1

                for token in sequence:
                    seq_token_pieces.append(self.tokenizer.tokenize(token))

                for token_pieces in seq_token_pieces:
                    seq_indices.append(seq_index)
                    seq_index += len(token_pieces)

                seq_token_pieces = [k for l in seq_token_pieces for k in l]
                seq_token_pieces.insert(0, "[CLS]")
                seq_token_pieces.append("[SEP]")

                all_token_pieces.append(seq_token_pieces)
                all_indices.append(seq_indices)

            max_seq_len_pieces = max([len(seq) for seq in all_token_pieces])
            max_seq_len_str = max([len(seq) for seq in sequence_list])

            tensor_all_input_ids = list()
            tensor_all_input_type_ids = list()
            tensor_all_input_mask = list()
            tensor_all_indices = list()

            for seq_token_pieces, seq_indices in zip(all_token_pieces, all_indices):
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

            all_encoders_layers, _ = self.model(
                tensor_all_input_ids,
                token_type_ids=tensor_all_input_type_ids,
                attention_mask=tensor_all_input_mask
            )

            selected_encoders_layers = list()
            for layer in all_encoders_layers:
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

        softmaxed_weights = F.softmax(self.weights, dim=0)
        weighted_sum = torch.mul(selected_encoders_layers[0], softmaxed_weights[0])

        for i, layer in enumerate(selected_encoders_layers[1:], start=1):
            weighted_sum = torch.add(weighted_sum, torch.mul(selected_encoders_layers[i], softmaxed_weights[i]))

        return weighted_sum
