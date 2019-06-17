import torch.nn as nn

from .crf import ConditionalRandomField
from .embedding import Embedder
from .lstm import LSTM


class LSTMCRF(nn.Module):

    def __init__(self,
                 constraints: list = None,
                 embedder: Embedder = None,
                 ffnn_hidden_layer_use: bool = None,
                 ffnn_hidden_layer_size: int = None,
                 ffnn_activation_function: str = None,
                 ffnn_input_dropout_rate: float = None,
                 input_size: int = None,
                 input_dropout_rate: float = None,
                 lstm_cell_size: int = None,
                 lstm_hidden_size: int = None,
                 lstm_layer_dropout_rate: int = None,
                 mappings: dict = None,
                 nb_layers: int = None,
                 num_labels: int = None,
                 skip_connections: bool = False):
        super().__init__()

        self.constraints = constraints
        self.embedder = embedder
        self.ffnn_hidden_layer_use = ffnn_hidden_layer_use
        self.ffnn_hidden_layer_size = ffnn_hidden_layer_size
        self.ffnn_activation_function = ffnn_activation_function
        self.ffnn_input_dropout_rate = ffnn_input_dropout_rate
        self.input_size = input_size
        self.input_dropout_rate = input_dropout_rate
        self.lstm_cell_size = lstm_cell_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layer_dropout_rate = lstm_layer_dropout_rate
        self.mappings = mappings
        self.nb_layers = nb_layers
        self.num_labels = num_labels
        self.skip_connections = skip_connections

        self.lstm_layer_input_size = self.lstm_hidden_size * 2 + self.input_size

        self.lstm_stack = self.create_lstm_stack()
        self.projection_layer = self.create_final_layer()

        self.crf: nn.Module = ConditionalRandomField(num_tags=self.num_labels, constraints=constraints)

    def create_final_layer(self):

        if self.ffnn_hidden_layer_use:
            if self.ffnn_hidden_layer_size == -1:
                current_ffnn_hidden_layer_size = self.lstm_hidden_size
            else:
                current_ffnn_hidden_layer_size = self.ffnn_hidden_layer_size

            module_list = list()
            module_list.append(nn.Linear(self.lstm_hidden_size * 2, current_ffnn_hidden_layer_size))

            if self.ffnn_activation_function == "relu":
                module_list.append(nn.ReLU())

            elif self.ffnn_activation_function == "tanh":
                module_list.append(nn.Tanh())

            else:
                raise Exception("The activation function is not supported: {}".format(self.ffnn_activation_function))

            module_list.append(nn.Dropout(p=self.ffnn_input_dropout_rate))
            module_list.append(nn.Linear(current_ffnn_hidden_layer_size, self.num_labels))

            projection_layer = nn.Sequential(*module_list)

        else:
            projection_layer = nn.Sequential(
                nn.Linear(self.lstm_hidden_size * 2, self.num_labels)
            )

        return projection_layer

    def create_lstm_stack(self):

        layer_idx = 0

        layers = nn.ModuleDict()
        if self.nb_layers == 1:
            skip_connection = False
        else:
            skip_connection = True

        layers[str(layer_idx)] = LSTM(lstm_hidden_size=self.lstm_hidden_size,
                                      lstm_cell_size=self.lstm_cell_size,
                                      input_dropout_rate=self.input_dropout_rate,
                                      input_size=self.input_size,
                                      skip_connection=skip_connection)
        layer_idx += 1

        while layer_idx < self.nb_layers:
            if layer_idx + 1 == self.nb_layers:
                skip_connection = False
            else:
                if self.skip_connections:
                    skip_connection = True
                else:
                    skip_connection = False

            layers[str(layer_idx)] = LSTM(lstm_hidden_size=self.lstm_hidden_size,
                                          lstm_cell_size=self.lstm_cell_size,
                                          input_dropout_rate=self.lstm_layer_dropout_rate,
                                          input_size=self.lstm_layer_input_size,
                                          skip_connection=skip_connection)
            layer_idx += 1

        return layers

    def forward(self, *args, **kwargs):

        cuda = kwargs["cuda"]
        batch = kwargs["batch"]

        if cuda:
            batch["chr"] = batch["chr"].cuda()
            batch["tok"] = batch["tok"].cuda()
            batch["elmo"] = batch["elmo"].cuda()

        batch_embed = self.embedder(batch, cuda)

        # Sorting batch by size (from
        batch_len_sort, batch_perm_idx = batch["tok_len"].sort(descending=True)
        batch_embed = batch_embed[batch_perm_idx]

        layer_output = None
        for layer_idx, layer in self.lstm_stack.items():
            if int(layer_idx) == 0:
                layer_output = layer(batch_embed, batch_embed, batch_len_sort)

            else:
                layer_output = layer(batch_embed, layer_output, batch_len_sort)

        # Sorting back the input to the original order
        _, r = batch_perm_idx.sort()
        layer_output = layer_output[r]

        logits = self.projection_layer(layer_output)

        return logits

    def get_loss(self, batch, cuda):

        logits = self.forward(batch=batch, cuda=cuda)

        if cuda:
            batch["labels"] = batch["labels"].cuda()
            batch["mask"] = batch["mask"].cuda()

        return -self.crf(logits, batch["labels"], mask=batch["mask"]), "loss/crf"

    def get_labels(self, batch, cuda):

        inverted_label_mapping = {v: k for k, v in self.mappings["ner_labels"].items()}

        if cuda:
            batch["labels"] = batch["labels"].cuda()
            batch["mask"] = batch["mask"].cuda()

        logits = self.forward(batch=batch, cuda=cuda)

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
            pred_converted.append([inverted_label_mapping.get(item) for item in path])

        for path in gs:
            gs_converted.append([inverted_label_mapping.get(item) for item in path])

        gs_converted = [item for seq in gs_converted for item in seq]
        pred_converted = [item for seq in pred_converted for item in seq]

        return {"pred": pred_converted, "gs": gs_converted}

    def infer_labels(self, batch, cuda):

        inverted_label_mapping = {v: k for k, v in self.mappings["ner_labels"].items()}

        if cuda:
            batch["mask"] = batch["mask"].cuda()

        logits = self.forward(batch=batch, cuda=cuda)
        best_paths = self.crf.viterbi_tags(logits, batch["mask"])
        pred = [best_path for best_path, _ in best_paths]
        pred_converted = list()

        for path in pred:
            pred_converted.append([inverted_label_mapping.get(item) for item in path])

        return pred_converted
