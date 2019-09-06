import torch
import torch.nn as nn
from allennlp.modules.augmented_lstm import AugmentedLstm
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class LSTMAugmented(nn.Module):

    def __init__(self,
                 lstm_hidden_size: int = None,
                 input_dropout_rate: float = None,
                 input_size: int = None,
                 use_highway: bool = False):
        super().__init__()

        self.lstm_hidden_size = lstm_hidden_size
        self.input_dropout_rate = input_dropout_rate
        self.input_size = input_size
        self.use_highway = use_highway

        self.lstm_forward = AugmentedLstm(self.input_size,
                                          self.lstm_hidden_size,
                                          go_forward=True,
                                          recurrent_dropout_probability=self.input_dropout_rate,
                                          use_highway=self.use_highway,
                                          use_input_projection_bias=False)

        self.lstm_backward = AugmentedLstm(self.input_size,
                                           self.lstm_hidden_size,
                                           go_forward=False,
                                           recurrent_dropout_probability=self.input_dropout_rate,
                                           use_highway=self.use_highway,
                                           use_input_projection_bias=False)

    def forward(self, batch_packed):

        out_forward, (_, _) = self.lstm_forward(batch_packed)
        out_backward, (_, _) = self.lstm_backward(batch_packed)

        forward_unpacked, forward_len = pad_packed_sequence(out_forward, batch_first=True)
        backward_unpacked, backward_len = pad_packed_sequence(out_backward, batch_first=True)

        output = torch.cat((forward_unpacked, backward_unpacked), 2)
        output = pack_padded_sequence(output, forward_len, batch_first=True)

        return output
