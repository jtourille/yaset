import torch
import torch.nn as nn
from allennlp.modules.lstm_cell_with_projection import LstmCellWithProjection


class LSTM(nn.Module):

    def __init__(self,
                 lstm_hidden_size: int = None,
                 lstm_cell_size: int = None,
                 input_dropout_rate: float = None,
                 input_size: int = None,
                 skip_connection: bool = False):
        super().__init__()

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_cell_size = lstm_cell_size
        self.input_dropout_rate = input_dropout_rate
        self.input_size = input_size
        self.skip_connection = skip_connection

        self.lstm_forward = LstmCellWithProjection(self.input_size, self.lstm_hidden_size,
                                                   self.lstm_cell_size,
                                                   go_forward=True,
                                                   recurrent_dropout_probability=self.input_dropout_rate)

        self.lstm_backward = LstmCellWithProjection(self.input_size, self.lstm_hidden_size,
                                                    self.lstm_cell_size,
                                                    go_forward=False,
                                                    recurrent_dropout_probability=self.input_dropout_rate)

        self.lstm_forward.reset_parameters()
        self.lstm_backward.reset_parameters()

    def forward(self, batch_embed, input_layer, batch_len):

        out_forward, (_, _) = self.lstm_forward(input_layer, batch_len)
        out_backward, (_, _) = self.lstm_backward(input_layer, batch_len)

        if self.skip_connection:
            output = torch.cat((out_forward, out_backward, batch_embed), 2)
        else:
            output = torch.cat((out_forward, out_backward), 2)

        return output
