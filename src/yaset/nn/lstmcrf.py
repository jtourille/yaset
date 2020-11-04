import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from yaset.nn.crf import ConditionalRandomField
from yaset.nn.embedding import Embedder
from yaset.nn.lstm import LSTMAugmented


# class VanillaLSTMCRF(nn.Module):
#
#     def __init__(self,
#                  constraints: list = None,
#                  embedder: Embedder = None,
#                  ffnn_hidden_layer_use: bool = None,
#                  ffnn_hidden_layer_size: int = None,
#                  ffnn_activation_function: str = None,
#                  ffnn_input_dropout_rate: float = None,
#                  input_size: int = None,
#                  lstm_hidden_size: int = None,
#                  lstm_input_dropout_rate: float = 0.0,
#                  lstm_layer_dropout_rate: float = 0.0,
#                  mappings: dict = None,
#                  nb_layers: int = None,
#                  num_labels: int = None):
#         super().__init__()
#
#         self.constraints = constraints
#         self.embedder = embedder
#         self.ffnn_hidden_layer_use = ffnn_hidden_layer_use
#         self.ffnn_hidden_layer_size = ffnn_hidden_layer_size
#         self.ffnn_activation_function = ffnn_activation_function
#         self.ffnn_input_dropout_rate = ffnn_input_dropout_rate
#         self.input_size = input_size
#         self.lstm_hidden_size = lstm_hidden_size
#         self.lstm_input_dropout_rate = lstm_input_dropout_rate
#         self.lstm_layer_dropout_rate = lstm_layer_dropout_rate
#         self.mappings = mappings
#         self.nb_layers = nb_layers
#         self.num_labels = num_labels
#
#         self.lstm_stack = nn.LSTM(input_size=self.input_size,
#                                   hidden_size=self.lstm_hidden_size,
#                                   num_layers=self.nb_layers,
#                                   bias=True,
#                                   batch_first=True,
#                                   dropout=self.lstm_layer_dropout_rate,
#                                   bidirectional=True)
#
#         self.projection_layer = self.create_final_layer()
#
#         if self.nb_layers == 0:
#             self.ensemble_output_size = self.input_size
#         else:
#             self.ensemble_output_size = self.lstm_hidden_size * 2
#
#         self.crf: nn.Module = ConditionalRandomField(num_tags=self.num_labels, constraints=constraints)
#
#     def create_final_layer(self):
#
#         if self.ffnn_hidden_layer_use:
#             if self.nb_layers == 0:
#                 current_input_projection = self.input_size
#             else:
#                 current_input_projection = self.lstm_hidden_size * 2
#
#             if self.ffnn_hidden_layer_size == -1:
#                 current_ffnn_hidden_layer_size = current_input_projection // 2
#             else:
#                 current_ffnn_hidden_layer_size = self.ffnn_hidden_layer_size
#
#             module_list = list()
#             module_list.append(nn.Linear(current_input_projection, current_ffnn_hidden_layer_size))
#
#             if self.ffnn_activation_function == "relu":
#                 module_list.append(nn.ReLU())
#
#             elif self.ffnn_activation_function == "tanh":
#                 module_list.append(nn.Tanh())
#
#             else:
#                 raise Exception("The activation function is not supported: {}".format(self.ffnn_activation_function))
#
#             module_list.append(nn.Dropout(p=self.ffnn_input_dropout_rate))
#             module_list.append(nn.Linear(current_ffnn_hidden_layer_size, self.num_labels))
#
#             projection_layer = nn.Sequential(*module_list)
#
#         else:
#             if self.nb_layers == 0:
#                 current_input_projection = self.input_size
#             else:
#                 current_input_projection = self.lstm_hidden_size * 2
#
#             projection_layer = nn.Sequential(
#                 nn.Linear(current_input_projection, self.num_labels)
#             )
#
#         return projection_layer
#
#     def forward(self, *args, **kwargs):
#
#         cuda = kwargs["cuda"]
#         batch = kwargs["batch"]
#
#         if cuda:
#             batch["chr_cnn"] = batch["chr_cnn"].cuda()
#             batch["chr_lstm"] = batch["chr_lstm"].cuda()
#             batch["tok"] = batch["tok"].cuda()
#             batch["elmo"] = batch["elmo"].cuda()
#
#         batch_embed = self.embedder(batch, cuda)
#
#         batch_len_sort, batch_perm_idx = batch["tok_len"].sort(descending=True)
#         batch_embed = batch_embed[batch_perm_idx]
#         batch_packed = pack_padded_sequence(batch_embed, batch_len_sort, batch_first=True)
#
#         output_packed, (_, _) = self.lstm_stack(batch_packed)
#         output_unpacked, output_len = pad_packed_sequence(output_packed, batch_first=True)
#
#         # Sorting back the input to the original order
#         _, r = batch_perm_idx.sort()
#         layer_output = output_unpacked[r]
#
#         return layer_output
#
#     def forward_ensemble_lstm(self, batch, cuda):
#
#         if cuda:
#             batch["chr_cnn"] = batch["chr_cnn"].cuda()
#             batch["chr_lstm"] = batch["chr_lstm"].cuda()
#             batch["tok"] = batch["tok"].cuda()
#             batch["elmo"] = batch["elmo"].cuda()
#
#         batch_embed = self.embedder(batch, cuda)
#
#         # Sorting batch by size (from
#         batch_len_sort, batch_perm_idx = batch["tok_len"].sort(descending=True)
#         batch_embed = batch_embed[batch_perm_idx]
#         batch_packed = pack_padded_sequence(batch_embed, batch_len_sort, batch_first=True)
#
#         output_packed, (_, _) = self.lstm_stack(batch_packed)
#         output_unpacked, output_len = pad_packed_sequence(output_packed, batch_first=True)
#
#         # Sorting back the input to the original order
#         _, r = batch_perm_idx.sort()
#         layer_output = output_unpacked[r]
#
#         layer_output = self.projection_layer(layer_output)
#
#         return layer_output
#
#     def get_loss(self, batch, cuda):
#
#         layer_output = self.forward(batch=batch, cuda=cuda)
#         logits = self.projection_layer(layer_output)
#
#         if cuda:
#             batch["labels"] = batch["labels"].cuda()
#             batch["mask"] = batch["mask"].cuda()
#
#         return -self.crf(logits, batch["labels"], mask=batch["mask"]), "loss/crf"
#
#     def get_loss_ensemble(self, batch, cuda):
#
#         layer_output = self.forward(batch=batch, cuda=cuda)
#         logits = self.projection_layer(layer_output)
#
#         if cuda:
#             batch["labels"] = batch["labels"].cuda()
#             batch["mask"] = batch["mask"].cuda()
#
#         return -self.crf(logits, batch["labels"], mask=batch["mask"]), layer_output
#
#     def get_labels(self, batch, cuda):
#
#         inverted_label_mapping = {v: k for k, v in self.mappings["ner_labels"].items()}
#
#         if cuda:
#             batch["labels"] = batch["labels"].cuda()
#             batch["mask"] = batch["mask"].cuda()
#
#         layer_output = self.forward(batch=batch, cuda=cuda)
#         logits = self.projection_layer(layer_output)
#
#         best_paths = self.crf.viterbi_tags(logits, batch["mask"])
#
#         pred = [best_path for best_path, _ in best_paths]
#
#         labels = batch["labels"].data.cpu().numpy().tolist()
#         lens = batch["tok_len"].data.cpu().numpy().tolist()
#
#         gs = list()
#         for seq, seq_len in zip(labels, lens):
#             gs.append(seq[:seq_len])
#
#         pred_converted = list()
#         gs_converted = list()
#
#         for path in pred:
#             pred_converted.append([inverted_label_mapping.get(item) for item in path])
#
#         for path in gs:
#             gs_converted.append([inverted_label_mapping.get(item) for item in path])
#
#         gs_converted = [item for seq in gs_converted for item in seq]
#         pred_converted = [item for seq in pred_converted for item in seq]
#
#         return {"pred": pred_converted, "gs": gs_converted}
#
#     def infer_labels(self, batch, cuda):
#
#         inverted_label_mapping = {v: k for k, v in self.mappings["ner_labels"].items()}
#
#         if cuda:
#             batch["mask"] = batch["mask"].cuda()
#
#         layer_output = self.forward(batch=batch, cuda=cuda)
#         logits = self.projection_layer(layer_output)
#         best_paths = self.crf.viterbi_tags(logits, batch["mask"])
#         pred = [best_path for best_path, _ in best_paths]
#         pred_converted = list()
#
#         for path in pred:
#             pred_converted.append([inverted_label_mapping.get(item) for item in path])
#
#         return pred_converted


class AugmentedLSTMCRF(nn.Module):
    def __init__(
        self,
        constraints: list = None,
        embedder: Embedder = None,
        ffnn_hidden_layer_use: bool = None,
        ffnn_hidden_layer_size: int = None,
        ffnn_activation_function: str = None,
        ffnn_input_dropout_rate: float = None,
        input_size: int = None,
        lstm_hidden_size: int = None,
        lstm_input_dropout_rate: float = None,
        lstm_layer_dropout_rate: int = None,
        mappings: dict = None,
        nb_layers: int = None,
        num_labels: int = None,
        use_highway: bool = False,
    ):
        super().__init__()

        self.constraints = constraints
        self.embedder = embedder
        self.ffnn_hidden_layer_use = ffnn_hidden_layer_use
        self.ffnn_hidden_layer_size = ffnn_hidden_layer_size
        self.ffnn_activation_function = ffnn_activation_function
        self.ffnn_input_dropout_rate = ffnn_input_dropout_rate
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_input_dropout_rate = lstm_input_dropout_rate
        self.lstm_layer_dropout_rate = lstm_layer_dropout_rate
        self.mappings = mappings
        self.nb_layers = nb_layers
        self.num_labels = num_labels
        self.use_highway = use_highway

        self.lstm_stack = self.create_lstm_stack()
        self.projection_layer = self.create_final_layer()

        if self.nb_layers == 0:
            self.ensemble_output_size = self.input_size
        else:
            self.ensemble_output_size = self.lstm_hidden_size * 2

        self.crf: nn.Module = ConditionalRandomField(
            num_tags=self.num_labels, constraints=constraints
        )

    def create_final_layer(self):

        if self.ffnn_hidden_layer_use:
            if self.nb_layers == 0:
                current_input_projection = self.input_size
            else:
                current_input_projection = self.lstm_hidden_size * 2

            if self.ffnn_hidden_layer_size == -1:
                current_ffnn_hidden_layer_size = current_input_projection // 2
            else:
                current_ffnn_hidden_layer_size = self.ffnn_hidden_layer_size

            module_list = list()
            module_list.append(
                nn.Linear(
                    current_input_projection, current_ffnn_hidden_layer_size
                )
            )

            if self.ffnn_activation_function == "relu":
                module_list.append(nn.ReLU())

            elif self.ffnn_activation_function == "tanh":
                module_list.append(nn.Tanh())

            else:
                raise Exception(
                    "The activation function is not supported: {}".format(
                        self.ffnn_activation_function
                    )
                )

            module_list.append(nn.Dropout(p=self.ffnn_input_dropout_rate))
            module_list.append(
                nn.Linear(current_ffnn_hidden_layer_size, self.num_labels)
            )

            projection_layer = nn.Sequential(*module_list)

        else:
            if self.nb_layers == 0:
                current_input_projection = self.input_size
            else:
                current_input_projection = self.lstm_hidden_size * 2

            projection_layer = nn.Sequential(
                nn.Linear(current_input_projection, self.num_labels)
            )

        return projection_layer

    def create_lstm_stack(self):

        layers = nn.ModuleDict()

        if self.nb_layers > 0:
            layer_idx = 0

            layers[str(layer_idx)] = LSTMAugmented(
                lstm_hidden_size=self.lstm_hidden_size,
                input_dropout_rate=self.lstm_input_dropout_rate,
                input_size=self.input_size,
                use_highway=self.use_highway,
            )
            layer_idx += 1

            while layer_idx < self.nb_layers:
                layers[str(layer_idx)] = LSTMAugmented(
                    lstm_hidden_size=self.lstm_hidden_size,
                    input_dropout_rate=self.lstm_layer_dropout_rate,
                    input_size=self.lstm_hidden_size * 2,
                    use_highway=self.use_highway,
                )
                layer_idx += 1

        return layers

    def forward(self, *args, **kwargs):

        cuda = kwargs["cuda"]
        batch = kwargs["batch"]

        if cuda:
            batch["chr_cnn_literal"] = batch["chr_cnn_literal"].cuda()
            batch["chr_cnn_utf8"] = batch["chr_cnn_utf8"].cuda()
            # batch["chr_lstm_type1"] = batch["chr_lstm_type1"].cuda()
            # batch["chr_lstm_type2"] = batch["chr_lstm_type2"].cuda()
            batch["tok"] = batch["tok"].cuda()
            batch["elmo"] = batch["elmo"].cuda()

        batch_embed = self.embedder(batch, cuda)

        # Sorting batch by size (from
        batch_len_sort, batch_perm_idx = batch["tok_len"].sort(descending=True)
        batch_embed = batch_embed[batch_perm_idx]
        batch_packed = pack_padded_sequence(
            batch_embed, batch_len_sort, batch_first=True
        )

        layer_output = batch_packed
        for layer_idx, layer in self.lstm_stack.items():
            layer_output = layer(layer_output)

        layer_output, _ = pad_packed_sequence(layer_output, batch_first=True)

        # Sorting back the input to the original order
        _, r = batch_perm_idx.sort()
        layer_output = layer_output[r]

        return layer_output

    # def forward_ensemble_lstm(self, batch, cuda):
    #
    #     if cuda:
    #         batch["chr_cnn"] = batch["chr_cnn"].cuda()
    #         batch["chr_lstm"] = batch["chr_lstm"].cuda()
    #         batch["tok"] = batch["tok"].cuda()
    #         batch["elmo"] = batch["elmo"].cuda()
    #
    #     batch_embed = self.embedder(batch, cuda)
    #
    #     # Sorting batch by size (from
    #     batch_len_sort, batch_perm_idx = batch["tok_len"].sort(descending=True)
    #     batch_embed = batch_embed[batch_perm_idx]
    #     batch_packed = pack_padded_sequence(batch_embed, batch_len_sort, batch_first=True)
    #
    #     layer_output = batch_packed
    #     for layer_idx, layer in self.lstm_stack.items():
    #         layer_output = layer(layer_output)
    #
    #     layer_output, _ = pad_packed_sequence(layer_output, batch_first=True)
    #
    #     # Sorting back the input to the original order
    #     _, r = batch_perm_idx.sort()
    #     layer_output = layer_output[r]
    #
    #     layer_output = self.projection_layer(layer_output)
    #
    #     return layer_output

    def forward_ensemble_lstm_attention(self, batch, cuda):

        if cuda:
            batch["chr_cnn_type1"] = batch["chr_cnn_type1"].cuda()
            batch["chr_cnn_type2"] = batch["chr_cnn_type2"].cuda()
            batch["chr_lstm"] = batch["chr_lstm"].cuda()
            batch["tok"] = batch["tok"].cuda()
            batch["elmo"] = batch["elmo"].cuda()

        batch_embed = self.embedder(batch, cuda)

        # Sorting batch by size (from
        batch_len_sort, batch_perm_idx = batch["tok_len"].sort(descending=True)
        batch_embed = batch_embed[batch_perm_idx]
        batch_packed = pack_padded_sequence(
            batch_embed, batch_len_sort, batch_first=True
        )

        layer_output = batch_packed
        for layer_idx, layer in self.lstm_stack.items():
            layer_output = layer(layer_output)

        layer_output, _ = pad_packed_sequence(layer_output, batch_first=True)

        # Sorting back the input to the original order
        _, r = batch_perm_idx.sort()
        layer_output = layer_output[r]

        return layer_output

    def get_loss(self, batch, cuda: bool = False):

        layer_output = self.forward(batch=batch, cuda=cuda)
        logits = self.projection_layer(layer_output)

        if cuda:
            batch["labels"] = batch["labels"].cuda()
            batch["mask"] = batch["mask"].cuda()

        return (
            -self.crf(logits, batch["labels"], mask=batch["mask"]),
            "loss/crf",
        )

    def get_loss_ensemble(self, batch, cuda):

        layer_output = self.forward(batch=batch, cuda=cuda)
        logits = self.projection_layer(layer_output)

        if cuda:
            batch["labels"] = batch["labels"].cuda()
            batch["mask"] = batch["mask"].cuda()

        return (
            -self.crf(logits, batch["labels"], mask=batch["mask"]),
            layer_output,
        )

    def get_labels(self, batch, cuda):

        inverted_label_mapping = {
            v: k for k, v in self.mappings["lbls"].items()
        }

        if cuda:
            batch["labels"] = batch["labels"].cuda()
            batch["mask"] = batch["mask"].cuda()

        layer_output = self.forward(batch=batch, cuda=cuda)
        logits = self.projection_layer(layer_output)
        best_paths = self.crf.viterbi_tags(logits, batch["mask"])

        pred = [best_path for best_path, _ in best_paths]

        labels = batch["labels"].data.cpu().numpy().tolist()
        lens = batch["tok_len"].data.cpu().numpy().tolist()

        gs = list()
        for seq, seq_len in zip(labels, lens):
            gs.append(seq[:seq_len])

        pred_converted = list()
        gs_converted = list()

        for path in pred:
            pred_converted.append(
                [inverted_label_mapping.get(item) for item in path]
            )

        for path in gs:
            gs_converted.append(
                [inverted_label_mapping.get(item) for item in path]
            )

        gs_converted = [item for seq in gs_converted for item in seq]
        pred_converted = [item for seq in pred_converted for item in seq]

        return {"pred": pred_converted, "gs": gs_converted}

    def infer_labels(self, batch, cuda):

        inverted_label_mapping = {
            v: k for k, v in self.mappings["lbls"].items()
        }

        if cuda:
            batch["mask"] = batch["mask"].cuda()

        layer_output = self.forward(batch=batch, cuda=cuda)
        logits = self.projection_layer(layer_output)
        best_paths = self.crf.viterbi_tags(logits, batch["mask"])

        pred = [best_path for best_path, _ in best_paths]
        pred_converted = list()

        for path in pred:
            pred_converted.append(
                [inverted_label_mapping.get(item) for item in path]
            )

        return pred_converted

    # def dev_infer_labels(self, batch, cuda):
    #
    #     inverted_label_mapping = {v: k for k, v in self.mappings["ner_labels"].items()}
    #
    #     if cuda:
    #         batch["mask"] = batch["mask"].cuda()
    #
    #     layer_output = self.forward(batch=batch, cuda=cuda)
    #     logits = self.projection_layer(layer_output)
    #     best_paths = self.crf.viterbi_tags(logits, batch["mask"])
    #     pred, loss = zip(*best_paths)
    #     pred_converted = list()
    #
    #     for path in pred:
    #         pred_converted.append([inverted_label_mapping.get(item) for item in path])
    #
    #     return pred_converted, loss
